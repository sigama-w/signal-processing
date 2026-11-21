"""数据集读取（完全对齐官方真实数据格式）

官方训练数据 schema (rf_signal_benchmark_train_v1.0, 19 列):
  task: protocol / modulation / parameter        （注意: 大表按行混排后分块，
  sample_id: 唯一标识                              必须按 task 列过滤，不能按文件假设）
  iq: list<list<float>>  [N,2] I/Q 时序
  modulation_label / modulation_valid             （task=protocol 行为 None）
  protocol_label  / protocol_valid                （task=modulation 行为 None）
  symbol_rate ... number_of_pulses                （7 参数列, 无前缀, SI 单位）
  symbol_rate_valid ... number_of_pulses_valid     （不适用参数真值 = 0.0）

  官方测试输入 (sample_io) 仅 3 列: task / sample_id / iq（混合三任务）

兼容性:
  - iq 双格式: 官方原生 list<list<float>> / 本框架模拟数据的扁平 bytes
  - 列名双风格: 官方 (modulation_label 等) / 旧模拟 (label / param_* / valid_*)
  - 任务名: protocol（官方）; 旧 "system" 已废弃

归一化策略:
  - IQ: 模型内部按样本峰值归一化（tokenizer.normalize），此处不动
  - 回归参数: min-max 归一化到 [0,1]，无效维目标置 0，推理时反归一化
"""

import os
import io
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

PARAM_NAMES = [
    "symbol_rate", "bandwidth", "frequency_shift",
    "pulse_width", "pri", "time_delay", "number_of_pulses",
]

# ---- 官方类别表（task_schema.json 顺序，唯一可信来源）----
MODULATION_CLASSES = [
    "OOK", "BPSK", "QPSK", "8PSK", "16QAM", "32QAM", "64QAM", "256QAM",
    "OQPSK", "2FSK", "4FSK", "MSK", "128QAM", "AQPSK", "CSS", "GFSK",
    "pi_4DQPSK", "FM",
    "Radar/Coherent Pulse Train", "Radar/Barker Code",
    "Radar/Polyphase Barker Code", "Radar/Frank Code", "Radar/LFM",
]
PROTOCOL_CLASSES = [
    "WiFi", "Bluetooth", "LoRa", "5G", "ZigBee",
    "FM broadcast", "Radar", "Idle background",
]
# 向后兼容别名（旧代码引用）
SYSTEM_CLASSES = PROTOCOL_CLASSES

LABEL2ID = {
    "protocol": {c: i for i, c in enumerate(PROTOCOL_CLASSES)},
    "modulation": {c: i for i, c in enumerate(MODULATION_CLASSES)},
}
ID2LABEL = {t: {i: c for c, i in m.items()} for t, m in LABEL2ID.items()}

# ---- 官方全量统计 (1,179,000 行, docs/real_data_stats.json, SI 单位) ----
DEFAULT_PARAM_STATS = {
    "min": np.array([62.5, 65.0, -50.0,
                     1.0000078873417806e-05, 1.7000029401970096e-05,
                     1.000015004137822e-06, 2.0], dtype=np.float32),
    "max": np.array([2000.0, 3500.0, 1000.0,
                     1.5999929019017145e-05, 2.2999996872385964e-05,
                     9.999991561926436e-06, 6.0], dtype=np.float32),
}

FS = 1_000_000.0  # 默认采样率（官方数据无采样率字段）


def to_iq_array(v, max_len=None):
    """双格式 iq 读取 -> [N,2] float32

    官方原生列 (list<list<float>> / ndarray of ndarray) 或本框架扁平 bytes 均支持。
    max_len 不为 None 时截断/右侧补零。
    """
    if isinstance(v, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(v, dtype=np.float32).reshape(-1, 2).copy()
    else:
        arr = np.asarray(v)
        if arr.dtype == object:  # pandas 读原生嵌套列的 object 行
            arr = np.asarray(arr.tolist(), dtype=np.float32)
        else:
            arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"unexpected iq shape {arr.shape}")
    if max_len is not None:
        if arr.shape[0] > max_len:
            arr = arr[:max_len]
        elif arr.shape[0] < max_len:
            arr = np.pad(arr, ((0, max_len - arr.shape[0]), (0, 0)))
    return arr


# 向后兼容别名
def bytes_to_iq(b, max_len=None):
    return to_iq_array(b, max_len)


def _detect_style(df):
    """检测列名风格: 'official' 或 'legacy'(模拟数据)"""
    official_marks = ("modulation_label", "protocol_label", "symbol_rate_valid")
    return "official" if any(c in df.columns for c in official_marks) else "legacy"


class EMSignalDataset(Dataset):
    """单任务数据集（一个 Parquet 文件 -> 一个 Dataset）

    自动适配官方 / 模拟两种列名风格；parameter 任务额外返回 pattern 标签
    （0=通信模式: symbol_rate/bandwidth/frequency_shift 有效
      1=雷达模式: pulse_width/pri/time_delay/number_of_pulses 有效）
    """

    def __init__(self, parquet_path, task, seq_len=1024, param_stats=None):
        self.df = pd.read_parquet(parquet_path)
        self.task = "protocol" if task == "system" else task
        if "task" in self.df.columns:
            self.df = self.df[self.df["task"].replace({"system":"protocol"}) == self.task].reset_index(drop=True)
        self.seq_len = seq_len
        self.param_stats = param_stats
        self.style = _detect_style(self.df)
        if self.style == "official":
            self._label_col = "modulation_label" if task == "modulation" else "protocol_label"
            self._param_cols = PARAM_NAMES
            self._valid_cols = [n + "_valid" for n in PARAM_NAMES]
        else:
            self._label_col = "label"
            self._param_cols = [f"param_{n}" for n in PARAM_NAMES]
            self._valid_cols = [f"valid_{n}" for n in PARAM_NAMES]
        self.has_label = self._label_col in self.df.columns
        self.has_params = self._param_cols[0] in self.df.columns

    def __len__(self):
        return len(self.df)

    def _label_of(self, row):
        v = row[self._label_col]
        return None if v is None or pd.isna(v) else v

    def __getitem__(self, i):
        row = self.df.iloc[i]
        iq = to_iq_array(row["iq"], self.seq_len)  # [N,2]
        item = {
            "iq": torch.from_numpy(iq),
            "sr": torch.tensor(FS, dtype=torch.float32),
            "task": self.task,
            "sample_id": row["sample_id"],
        }
        if self.task in ("protocol", "modulation") and self.has_label:
            lab = self._label_of(row)
            item["label"] = torch.tensor(LABEL2ID[self.task][lab], dtype=torch.long)
        if self.task == "parameter" and self.has_params:
            params = np.array([row[c] for c in self._param_cols], dtype=np.float32)
            valid = np.array([bool(row[c]) for c in self._valid_cols], dtype=np.float32)
            item["raw_params"] = torch.from_numpy((params * valid).astype(np.float32))
            # pattern: 通信(前3参数有效)=0 / 雷达(后4参数有效)=1
            pattern = float(valid[3])  # pulse_width_valid
            if self.param_stats is not None:
                mn, mx = self.param_stats["min"], self.param_stats["max"]
                params = (params - mn) / (mx - mn + 1e-12)
                params = np.clip(params, 0.0, 1.0)
            # 无效维目标置 0（原始真值不适用项 = 0.0；pattern 头推理时硬掩码）
            params = params * valid
            item["params"] = torch.from_numpy(params.astype(np.float32))
            item["valid"] = torch.from_numpy(valid)
            item["pattern"] = torch.tensor(int(pattern), dtype=torch.long)
        return item


class PreparedOfficialDataset(Dataset):
    """官方数据预处理后的快速数据集（memmap 版，配合 utils/prepare_official.py）

    目录结构:
      data_dir/{task}/iq.memap   —— float32 [total, seq_len, 2]（每个样本对齐到 seq_len）
      data_dir/{task}/meta.json  —— {"seq_len": N, "labels": [...], "params": [[...]],
                                     "valid": [[...]], "patterns": [...], "sample_ids": [...],
                                     "val_sample_ids": [...], "test_sample_ids": [...]}
    """

    def __init__(self, data_dir, task, seq_len=1024, param_stats=None):
        d = os.path.join(data_dir, task)
        with open(os.path.join(d, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.meta["seq_len"] != seq_len:
            raise ValueError(f"{task} meta seq_len={self.meta['seq_len']} != {seq_len}")
        self.task = task
        self.seq_len = seq_len
        self.param_stats = param_stats
        self.n = len(self.meta["sample_ids"])
        self.iq = np.memmap(os.path.join(d, "iq.memap"), dtype=np.float32, mode="r",
                            shape=(self.n, seq_len, 2))

    def __len__(self):
        return self.n

    def lengths(self, chunk=8192):
        """每个样本的原始长度（512/1024，按后半段是否全零判断），首次计算后缓存 lengths.npy。"""
        d = os.path.dirname(self.iq.filename)
        cache = os.path.join(d, "lengths.npy")
        if os.path.exists(cache):
            arr = np.load(cache)
            if len(arr) == self.n:
                return arr
        half = self.seq_len // 2
        out = np.empty(self.n, dtype=np.int32)
        for s in range(0, self.n, chunk):
            blk = np.asarray(self.iq[s:s + chunk, half:, :])
            out[s:s + chunk] = np.where(np.any(blk != 0, axis=(1, 2)), self.seq_len, half)
        try:
            np.save(cache, out)
        except OSError:
            pass
        return out

    def __getitem__(self, i):
        iq = np.array(self.iq[i])  # memmap 切片拷贝
        item = {
            "iq": torch.from_numpy(iq),
            "sr": torch.tensor(FS, dtype=torch.float32),
            "task": self.task,
            "sample_id": self.meta["sample_ids"][i],
        }
        if self.task in ("protocol", "modulation"):
            item["label"] = torch.tensor(LABEL2ID[self.task][self.meta["labels"][i]],
                                         dtype=torch.long)
        else:
            params = np.array(self.meta["params"][i], dtype=np.float32)
            valid = np.array(self.meta["valid"][i], dtype=np.float32)
            pattern = int(self.meta["patterns"][i])
            # v2：原始 SI 参数（无效位置 0），由 ParamCodec 在 GPU 上编码
            item["raw_params"] = torch.from_numpy((params * valid).astype(np.float32))
            if self.param_stats is not None:
                mn, mx = self.param_stats["min"], self.param_stats["max"]
                params = np.clip((params - mn) / (mx - mn + 1e-12), 0.0, 1.0)
            params = params * valid
            item["params"] = torch.from_numpy(params.astype(np.float32))
            item["valid"] = torch.from_numpy(valid)
            item["pattern"] = torch.tensor(int(pattern), dtype=torch.long)
        return item


def make_task_dataset(data_dir, task, split="train", seq_len=1024, param_stats=None):
    """统一工厂：优先用预处理 memmap（官方数据），否则退回 Parquet。

    data_dir 布局二选一:
      官方预处理: {task}/iq.memap + {task}/meta.json（val/test 划分记录在 meta 的
                  val_sample_ids / test_sample_ids，train/val/test 均从同一 memmap 过滤）
      模拟数据:   {split}_{task}.parquet（data_gen.py 产出）
    """
    mm_dir = os.path.join(data_dir, task)
    has_mm = os.path.exists(os.path.join(mm_dir, "meta.json"))
    if has_mm:
        full = PreparedOfficialDataset(data_dir, task, seq_len, param_stats)
        val_ids = set(full.meta.get("val_sample_ids", []))
        test_ids = set(full.meta.get("test_sample_ids", []))
        if split == "val" and not val_ids:
            raise FileNotFoundError(f"{task} has no val split in meta")
        if split == "test" and not test_ids:
            raise FileNotFoundError(
                f"{task} has no test split in meta; re-run utils/prepare_official.py "
                "(a recent version writes test_sample_ids)")
        if split == "test":
            keep = [i for i, s in enumerate(full.meta["sample_ids"]) if s in test_ids]
        else:
            # train = 全量减 val 减 test；val = val 集合（test 与 val 互斥）
            keep = [i for i, s in enumerate(full.meta["sample_ids"])
                    if (s in val_ids) == (split == "val") and s not in test_ids]
        return torch.utils.data.Subset(full, keep)
    pq_path = os.path.join(data_dir, f"{split}_{task}.parquet")
    if os.path.exists(pq_path):
        return EMSignalDataset(pq_path, task, seq_len, param_stats)
    # 兼容旧模拟数据命名：protocol 任务旧名为 system（EMSignalDataset 内部已做映射）
    if task == "protocol":
        legacy_path = os.path.join(data_dir, f"{split}_system.parquet")
        if os.path.exists(legacy_path):
            return EMSignalDataset(legacy_path, task, seq_len, param_stats)
    raise FileNotFoundError(f"no dataset for task={task}, split={split} under {data_dir}")


def collate_fn(batch):
    """自定义 collate：iq 堆叠为 [B,N,2]，其余字段对齐"""
    out = {
        "iq": torch.stack([b["iq"] for b in batch]),
        "sr": torch.stack([b["sr"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
        "task": batch[0]["task"],
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    if "params" in batch[0]:
        out["params"] = torch.stack([b["params"] for b in batch])
        out["valid"] = torch.stack([b["valid"] for b in batch])
        out["pattern"] = torch.stack([b["pattern"] for b in batch])
    if "raw_params" in batch[0]:
        out["raw_params"] = torch.stack([b["raw_params"] for b in batch])
    return out


def compute_param_stats(parquet_path):
    """参数归一化统计。官方风格直接返回全量统计常量（避免重扫 GB 级文件）"""
    df_cols = pd.read_parquet(parquet_path).columns
    if "symbol_rate_valid" in df_cols:
        return {"min": DEFAULT_PARAM_STATS["min"].copy(),
                "max": DEFAULT_PARAM_STATS["max"].copy()}
    df = pd.read_parquet(parquet_path)
    mins, maxs = [], []
    for n in PARAM_NAMES:
        col = n if n in df.columns else f"param_{n}"
        flag = n+"_valid" if n+"_valid" in df.columns else f"valid_{n}"
        v = df.loc[df[flag].astype(bool), col].to_numpy(dtype=np.float64)
        if not len(v):
            raise ValueError(f"No valid observations for {n}")
        mins.append(float(v.min())); maxs.append(float(v.max()))
    return {"min": np.array(mins, dtype=np.float32), "max": np.array(maxs, dtype=np.float32)}


def build_task_dataloaders(data_dir, batch_size=32, seq_len=1024, splits=("train", "val"),
                           tasks=("protocol", "modulation", "parameter"), num_workers=0,
                           rank=0, world_size=1):
    """构建多任务 DataLoader 字典

    优先使用预处理 memmap 布局（{task}/iq.memap + meta.json，val/test 划分在 meta 内），
    否则退回 Parquet 布局（{split}_{task}.parquet，模拟数据 data_gen.py 产出）。
    splits 可含 "test"（仅供训练后无偏评估，见 finetune_multitask.py --eval_test）。

    多卡（world_size > 1）时 train/val 划分均用 DistributedSampler 分片（batch_size
    为单卡批量，总批量 = batch_size × world_size）：train 分片训练，val 分片并行
    评估后 all-reduce 汇总指标；test 不分片（仅主卡训练后无偏评估用）。

    Returns:
        loaders = {
          "train": {"protocol": DataLoader, "modulation": ..., "parameter": ...},
          "val": {...},
        }, param_stats
    """
    stats_file = os.path.join(data_dir, "param_stats.json")
    if os.path.exists(stats_file):
        param_stats = load_param_stats(stats_file)
    else:
        param_stats = {"min": DEFAULT_PARAM_STATS["min"].copy(),
                       "max": DEFAULT_PARAM_STATS["max"].copy()}
    loaders = {}
    for split in splits:
        loaders[split] = {}
        for task in tasks:
            try:
                ds = make_task_dataset(data_dir, task, split=split, seq_len=seq_len,
                                       param_stats=param_stats if task == "parameter" else None)
            except FileNotFoundError:
                continue
            if world_size > 1:
                # train 打乱分片训练；val 跨卡分片并行评估（不打乱、strided 均匀），
                # 指标在 trainer 侧 all-reduce 汇总，避免主卡独自评估全量 val
                # 耗时过长、其余卡在集合通信上空等导致 HCCL 链路超时
                sampler = DistributedSampler(
                    ds, num_replicas=world_size, rank=rank,
                    shuffle=(split == "train"))
                loaders[split][task] = DataLoader(
                    ds, batch_size=batch_size, sampler=sampler,
                    collate_fn=collate_fn, num_workers=num_workers)
            else:
                loaders[split][task] = DataLoader(
                    ds, batch_size=batch_size, shuffle=(split == "train"),
                    collate_fn=collate_fn, num_workers=num_workers,
                )
    return loaders, param_stats


def save_param_stats(stats, path):
    """保存参数归一化统计（推理时需要反归一化）"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: np.asarray(v).tolist() for k, v in stats.items()}, f)


def load_param_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {"min": np.array(d["min"], dtype=np.float32),
            "max": np.array(d["max"], dtype=np.float32)}


# ============================================================ v2 数据加载增强
def dataset_lengths(ds):
    """返回数据集（含 Subset）每个样本的原始长度数组；无法获得时返回 None。"""
    if isinstance(ds, torch.utils.data.Subset):
        base = dataset_lengths(ds.dataset)
        return None if base is None else base[np.asarray(ds.indices)]
    if isinstance(ds, PreparedOfficialDataset):
        return ds.lengths()
    if isinstance(ds, EMSignalDataset):
        return np.array([len(to_iq_array(v)) for v in ds.df["iq"]], dtype=np.int32)
    return None


class LengthGroupedBatchSampler(torch.utils.data.Sampler):
    """按长度分桶组 batch（512 点与 1024 点样本不混批）。

    配合模型的 trim_padding：512 点 batch 时域 token 从 128 裁到 64，
    主干计算量约减半（40% 样本受益），训练整体提速 ~20-30%，且无补零噪声。
    每个 epoch：桶内打乱 -> 切 batch -> batch 顺序全局打乱。
    """

    def __init__(self, lengths, batch_size, shuffle=True, drop_last=True, seed=42):
        self.lengths = np.asarray(lengths)
        self.bs = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.groups = [np.where(self.lengths == v)[0] for v in np.unique(self.lengths)]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for g in self.groups:
            idx = rng.permutation(g) if self.shuffle else g
            for s in range(0, len(idx), self.bs):
                b = idx[s:s + self.bs]
                if len(b) < self.bs and self.drop_last:
                    continue
                batches.append(b.tolist())
        if self.shuffle:
            order = rng.permutation(len(batches))
            batches = [batches[i] for i in order]
        return batches

    def __iter__(self):
        batches = self._batches()
        self.epoch += 1          # 自动推进：每次重新迭代得到不同打乱顺序
        return iter(batches)

    def __len__(self):
        n = 0
        for g in self.groups:
            n += len(g) // self.bs if self.drop_last else -(-len(g) // self.bs)
        return n


def make_loader(ds, batch_size, train, num_workers=0, bucket=False, collate=collate_fn,
                seed=42, rank=0, world_size=1):
    """统一 DataLoader 构造：train 时 drop_last；单卡可按长度分桶；多卡用 DistributedSampler。"""
    common = dict(collate_fn=collate, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available(),
                  persistent_workers=num_workers > 0)
    if num_workers > 0:
        common["prefetch_factor"] = 4
    if world_size > 1:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=train)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=train, **common)
    if bucket:
        lengths = dataset_lengths(ds)
        if lengths is not None:
            bs = LengthGroupedBatchSampler(lengths, batch_size, shuffle=train,
                                           drop_last=train, seed=seed)
            return DataLoader(ds, batch_sampler=bs, **common)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, **common)


def build_task_dataloaders_v2(data_dir, batch_size=256, seq_len=1024, splits=("train", "val"),
                              tasks=("protocol", "modulation", "parameter"), num_workers=4,
                              bucket=True, eval_batch_size=None, rank=0, world_size=1, seed=42):
    """v2 多任务 loader：train 分桶 + drop_last；val/test 也分桶（推理更快且结果与不分桶一致）。"""
    loaders = {}
    for split in splits:
        loaders[split] = {}
        for task in tasks:
            try:
                ds = make_task_dataset(data_dir, task, split=split, seq_len=seq_len)
            except FileNotFoundError:
                continue
            bs = batch_size if split == "train" else (eval_batch_size or batch_size)
            loaders[split][task] = make_loader(ds, bs, split == "train", num_workers,
                                               bucket=bucket, seed=seed, rank=rank,
                                               world_size=world_size)
    return loaders


def load_param_arrays(data_dir, split="train"):
    """读取 parameter 任务原始 SI 参数与有效位（拟合 ParamCodec 用，只用 train 划分）。"""
    meta_path = os.path.join(data_dir, "parameter", "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        excl = set(meta.get("val_sample_ids", [])) | set(meta.get("test_sample_ids", []))
        keep = [i for i, s in enumerate(meta["sample_ids"]) if s not in excl] \
            if split == "train" else list(range(len(meta["sample_ids"])))
        p = np.asarray(meta["params"], dtype=np.float64)[keep]
        v = np.asarray(meta["valid"], dtype=bool)[keep]
        return p, v
    pq_path = os.path.join(data_dir, f"{split}_parameter.parquet")
    ds = EMSignalDataset(pq_path, "parameter")
    p = np.stack([[ds.df.iloc[i][c] for c in ds._param_cols] for i in range(len(ds))])
    v = np.stack([[bool(ds.df.iloc[i][c]) for c in ds._valid_cols] for i in range(len(ds))])
    return p.astype(np.float64), v
