# 代码解读报告（v2，面向编程基础一般的读者）

> 阅读建议：先看第 1~3 节建立整体印象（不用看代码），再按第 4 节按文件阅读，遇到不熟的 PyTorch 写法查第 5 节。
> 每个张量都标了形状，`B`=batch 大小，`N`=采样点数（1024），`T`=时域 token 数（128 或 64），`d`=特征维（384）。

---

## 1. 一张图看懂整个项目

```
           ┌──────────────────────── 离线一次 ────────────────────────┐
官方 Parquet ──► utils/prepare_official.py ──► datasets/{task}/iq.memap + meta.json
(120 万行)                                       （补零到 1024、划分 train/val/test）
           └──────────────────────────────────────────────────────────┘

阶段 1 预训练（无标签）  code/training/pretrain_mae.py
    三任务全部 IQ ──► 随机遮住 50% 的时间块 ──► 模型猜被遮住的波形 ──► weights/mae_pretrained.pt

阶段 2 多任务微调        code/training/finetune_multitask.py
    加载预训练权重 ──► 每步取 protocol/modulation/parameter 各一个 batch
                   ──► 三个损失的梯度加在一起 ──► 更新一次 ──► 每 1500 步在 val 上评估
                   ──► 分数创新高就导出 weights/config.json + model.safetensors + param_codec.json

推理（评测程序调用）      code/infer.py
    test.parquet ──► 按任务和长度分组 ──► model.predict ──► out/pred.jsonl
```

模型本身（`EMFoundationV2`）由三段组成：

```
IQ [B,1024,2]
   │
   ▼  ① 多视角 Tokenizer（multiview.py + dsp.py）  —— “把信号翻译成一串词”
tokens [B, ~194, d]：[CLS][统计][时域×T][频谱×32][自相关×16][星座×16]
   │
   ▼  ② 共享主干 UnifiedEncoder（layers/conformer.py） —— “理解这串词”
ctx [B, ~194, d]
   │
   ├─► ③a protocol 头 ──► 8 类 logits
   ├─► ③b modulation 头 ──► 23 类 logits（+ 5 类调制族辅助）
   └─► ③c parameter 头 ──► 通信/雷达判别 + 7 个物理参数
```

赛题要求“共享权重 > 80%”：①② 是共享的（约 90% 参数），③ 是各任务私有的小头。

---

## 2. 目录导读（只列你需要关心的）

```
code/
├── infer.py                 推理入口（评测直接调用）
├── configs/
│   ├── default.json         v2 默认配置 ← 改超参数就改这里
│   ├── default_v1_legacy.json 旧版配置
│   ├── ablation/*.json      消融实验配置
│   └── grid_v2.json         规模网格
├── models/
│   ├── __init__.py          build_model：按 arch 选 v1/v2
│   ├── dsp.py               ★ 信号处理前端（无可学习参数）
│   ├── multiview.py         ★ 多视角 Tokenizer
│   ├── layers/conformer.py  ★ 主干层
│   ├── ops/fused.py         融合算子（加速）
│   ├── heads_v2.py          ★ 任务头
│   ├── foundation_v2.py     ★ 模型总装
│   └── (foundation.py 等)   v1 旧模型，未改动
├── training/
│   ├── pretrain_mae.py      阶段 1 入口
│   ├── finetune_multitask.py阶段 2 入口
│   ├── trainer.py           训练循环（v1 + v2）
│   ├── tasks_v2.py          ★ v2 损失与评估
│   └── common.py            配置读取、预训练权重加载
└── utils/
    ├── dataset.py           数据读取、长度分桶
    ├── param_codec.py       ★ 参数编码/解码
    ├── augment.py           数据增强
    ├── optim.py             优化器、学习率调度、EMA
    ├── metrics.py           准确率等指标
    └── bad_case_analysis.py 错例分析
utils/                       （仓库根目录下的工具脚本）
├── prepare_official.py      数据预处理
├── measure_vram.py          显存实测
├── benchmark_latency.py     时延测试
├── profile_model.py         算子耗时分析
├── export_onnx.py           导出 ONNX
└── fit_param_codec.py       查看/拟合参数编码
```

---

## 3. 一个样本的完整旅程（带形状）

以一条 512 点的 modulation 样本为例：

| 步骤 | 代码位置 | 发生了什么 | 形状 |
|---|---|---|---|
| 1 | `prepare_official.py` | 读出 512×2 的 IQ，右侧补 512 个 0 | `[1024, 2]` |
| 2 | `dataset.PreparedOfficialDataset.__getitem__` | 从 memmap 取出，配上标签 | `iq [1024,2]`, `label` 标量 |
| 3 | `LengthGroupedBatchSampler` | 把同为 512 点的样本凑成一个 batch | `iq [256,1024,2]` |
| 4 | `augment_iq` | 随机旋转相位、可能加噪声（补零处保持 0） | 不变 |
| 5 | `dsp.valid_mask_from_iq` | 发现后 512 点全是 0 → 有效长度 512 | `valid [256,1024]`, `length=512` |
| 6 | `dsp.SignalFrontEnd.forward` | 归一化、FFT、估频偏、校正、STFT、各种特征 | 见 4.2 节 |
| 7 | `ConvStem` | 6 通道 × 1024 点 → 每 8 点一个 token | `[256,128,384]` |
| 8 | `GatedChannelFusion` | 和对齐的 STFT 帧融合 | `[256,128,384]` |
| 9 | `trim_padding` | 后 64 个 token 全是补零 → 裁掉 | `[256,64,384]` |
| 10 | 拼接 | CLS+统计+时域64+频谱32+自相关16+星座16 | `[256,130,384]` |
| 11 | `UnifiedEncoder` | 10 层 conv/conformer/attn | `[256,130,384]` |
| 12 | `ClsHeadV2` | CLS 与注意力池化拼接 → MLP | `[256,23]` |
| 13 | `cross_entropy` | 与真值比较得到损失 | 标量 |

推理时第 12 步之后做 `argmax`，再用 `ID2LABEL` 把编号翻译成 `"64QAM"` 这样的字符串。

---

## 4. 逐文件讲解

### 4.1 `code/utils/dataset.py` —— 数据从哪里来

* `PreparedOfficialDataset`：用 `np.memmap` 把 10GB 的 IQ 文件“映射”到内存地址，**按需读取**，不会一次性占满内存。`__getitem__(i)` 返回一个字典：`iq`、`label`（分类）或 `params/raw_params/valid/pattern`（参数）。
  * `raw_params`：v2 新增，原始物理单位（Hz、秒、个），无效参数位置为 0。
  * `valid`：7 个 0/1，表示哪些参数对这条样本有意义（通信样本前 3 个为 1，雷达样本后 4 个为 1）。
  * `pattern`：0=通信，1=雷达。
* `lengths()`：判断每条样本原始长度（看后半段是否全 0），结果存成 `lengths.npy`，第二次就秒读。
* `make_task_dataset(data_dir, task, split)`：根据 `meta.json` 里记录的 val/test 编号，返回对应子集（`torch.utils.data.Subset`）。
* `LengthGroupedBatchSampler`：**决定每个 batch 由哪些样本组成**。它先把样本按长度分组，组内打乱后切成 batch，再把所有 batch 打乱顺序。好处：一个 batch 里长度一致，可以把补零部分整体裁掉。
* `build_task_dataloaders_v2`：为三个任务、train/val 两个划分各建一个 `DataLoader`，返回 `loaders["train"]["modulation"]` 这样的嵌套字典。
* `load_param_arrays`：只读 train 划分的参数真值，用来拟合 `ParamCodec`（不能用 val/test，否则信息泄漏）。

### 4.2 `code/models/dsp.py` —— 信号处理前端（最“雷达/通信”味的部分）

这里**没有任何需要训练的参数**，全是确定性计算，相当于把专家手工做的信号分析自动化。

* `valid_mask_from_iq`：找最后一个非零点 → 有效长度（向上取整到 8 的倍数）。
* `RealDFT`：用矩阵乘法算 DFT。复数 `x = xr + j·xi`，DFT 矩阵 `e^{-j2πkn/N} = cos − j·sin`，展开后实部、虚部各是两次实数矩阵乘：
  `Xr = xr@C + xi@S`，`Xi = xi@C − xr@S`。为什么不用 `torch.fft`？国产 NPU 和 ONNX 对 FFT/复数支持不好，而矩阵乘是所有芯片最擅长的运算。
* `_normalize`：除以有效部分的均方根，信号变为单位功率；原始功率另存（进统计特征）。
* `_spectrum`：功率谱 → 平滑 → 噪底（最低一半频点的平均）→ 占用掩码（比噪底高 6dB 的频点）。
* `_carrier_correct`（频偏校正）：
  1. 在占用频带内对频率做功率加权平均 → 估计载频偏移 `cfo`；
  2. 把频谱整体平移 `round(cfo·N)` 个频点（`torch.gather` 实现每个样本不同的平移量）；
  3. 带外频点置零（去掉带外噪声，等效提高信噪比）；
  4. 逆 DFT 回时域，得到校正后的信号 `xc`。
  校正后 QAM 星座不再是圆环，高阶统计量也更可分。
* `_stft`：把信号切成 32 点一帧、每 8 点滑动一次，正好得到 128 帧，**和 128 个时域 token 一一对齐**。
* `_logspec`：对 `xc²`、`xc⁴` 求谱。物理意义：BPSK 平方后、QPSK/QAM 四次方后会在零频出现一根很强的谱线，是判断调制阶数的经典特征。
* `_envelope_acf`：包络（幅度）的自相关。雷达脉冲串的 PRI 会在自相关上表现为等间隔的峰，峰的个数与脉冲数有关。用 2N 点 DFT 计算，避免“循环相关”。
* `_constellation`：把 `xc` 的每个采样点落到 32×32 的格子里计数（`scatter_add_`），得到星座密度图。
* `_stats`：16 个标量特征，最后用 `symlog = sign(x)·log(1+|x|)` 压缩数值范围。
* `forward(iq, sample_keep=None, power=None)`：上面全部串起来，返回一个字典。`sample_keep` 只在预训练时用：先把被遮住的采样点置 0，再计算所有全局特征，防止模型通过频谱“偷看答案”。
* `pretrain_targets`：预训练要猜的“标准答案”（完整信号的 patch 与 STFT）。

### 4.3 `code/models/multiview.py` —— 把信号变成 token 序列

* `ConvStem`：三层一维卷积，总步长 8，把 `[B,6,1024]` 变成 `[B,128,d]`。比 v1 的“每 8 点拉平后乘一个矩阵”更能保留 8 点之内的细节（比如脉冲沿）。
* `GatedChannelFusion`：`t' = LN(t + σ(W[t;s]) ⊙ s)`。`σ` 输出 0~1 的“阀门”，逐通道决定吸收多少 STFT 信息。阀门偏置初始化为 −1，即开始时偏向时域。
* `MultiViewTokenizer.forward`：
  1. 调前端得到所有视角；
  2. 时域：stem → 融合 STFT →（预训练时把被遮住的 token 换成 `mask_token`）→ 加位置编码；
  3. 计算每个样本的有效 token 数，裁剪掉整批都是补零的部分；
  4. 生成 CLS（若开启任务提示，加上任务嵌入）、统计 token、频谱 token、自相关 token、星座 token，每种加上“类型编码”；
  5. 拼成一条序列，同时生成 `pad`（True 表示该位置是补零，注意力要忽略）；
  6. `layout["time_slice"]` 记录时域 token 在序列中的起止位置，卷积模块要用。

### 4.4 `code/models/layers/conformer.py` —— 主干

* `MHSA`：多头自注意力。`qkv` 一个线性层同时算出 Q、K、V，然后调 `ops.attention`（自动选 FlashAttention 等最快实现）。
* `ConvModule`：Conformer 的卷积模块。只取出时域那一段 `x[:, s:e]` 做卷积，补零位置先清 0，最后用 `F.pad` 把结果放回原长度（非时域位置为 0）。
* 三种层：
  * `AttnBlock`：`x + 注意力(LN(x))`，再 `x + FFN(LN(x))`（标准 Transformer）。
  * `ConformerBlock`：半个 FFN → 注意力 → 卷积 → 半个 FFN → LN（“马卡龙”夹心结构）。
  * `ConvBlock`：卷积 → FFN（没有注意力，最快）。
* `DropPath`：训练时随机跳过整个残差分支，越深的层跳过概率越大，防止过拟合。
* `UnifiedEncoder`：按 `layer_types` 列表逐层构建，前向就是 for 循环逐层调用，最后一个 LayerNorm。

### 4.5 `code/models/ops/fused.py` —— 加速算子

同一个函数名（`attention`、`swiglu`），内部根据设备选实现：GPU 上用 PyTorch 自带的 FlashAttention 和手写 Triton 核，昇腾上用 `torch_npu` 的融合算子，其他情况用最朴素的写法。**第一次调用会拿小数据和朴素写法对比结果**，不一致就自动退回朴素写法，并在 `fused_status()` 里记下原因。不理解 Triton 代码没关系，它只是 `silu(gate) * up` 的加速版。

### 4.6 `code/models/heads_v2.py` —— 任务头

* `AttnPool`：每个任务有自己的可学习“问题向量” `q`，和序列中每个 token 算相似度 → softmax → 加权平均。不同任务会自己学会关注不同 token（比如调制关注星座 token）。
* `PooledFeature`：把 CLS 向量和池化结果拼接，得到 `[B, 2d]`。
* `ClsHeadV2`：`[B,2d]` → 若干层 Linear+GELU → 输出 logits；可选输出调制族 logits（`aux`）与对比学习投影（`proj`）。
* `ParamHeadV2`：同样的主体，最后三个输出：`pattern`（2 类）、`reg`（连续参数的标准化值）、`cls`（每个离散参数一组 logits）。

### 4.7 `code/models/foundation_v2.py` —— 总装

* `__init__`：建 tokenizer、backbone、三个任务头、预训练重建头（`pretrain_head`）。
* `shared_parameter_count / head_parameter_count / shared_ratio`：统计共享率（任务提示属于私有）。
* `deploy_state_dict`：导出权重时去掉只在预训练用的重建头。
* `encode`：tokenizer + backbone。
* `head_outputs(iq, task)`：encode + 对应任务头。
* `predict(iq, task)`：推理用。分类返回类别编号；参数先用 `codec.decode` 变回物理单位，再根据 pattern 把不适用的参数置 0。
* `forward_pretrain(iq)`：
  1. 算出完整信号的答案（patch 与 STFT）；
  2. `_block_mask` 随机选 50% 的“块”（每块 4 个 token）遮住；
  3. 用遮住后的信号前向；
  4. 用线性层从时域 token 预测原始 patch 和 STFT，**只在被遮住且不是补零的位置**计算均方误差。

### 4.8 `code/utils/param_codec.py` —— 参数编码器

问题：7 个参数单位不同（Hz、秒、个数），数值差 10⁸ 倍，有的是整数，有的是长尾分布。

* `fit`：对每个参数看训练集中有效值的分布：
  * 不同取值 ≤ 32 个 → `cls`（当分类做）；
  * 都是正数且最大/最小 ≥ 20 → `log_zscore`（先取 log 再标准化）；
  * 其他 → `zscore`（减均值除标准差）。
* `encode(raw)`：物理值 → 网络要拟合的目标（`z` 和类别编号）。
* `decode(out)`：网络输出 → 物理值。分类参数用“概率 × 取值”求和（期望），这在均方误差意义下最优。
* `save/load`：存成 `param_codec.json`，推理时必须和权重一起提供（也已内嵌在 `config.json`）。

### 4.9 `code/utils/augment.py` —— 数据增强

* 相位旋转：`(I,Q)` 乘以旋转矩阵，等价于载波初相改变，所有标签都不变。
* 加噪声：按随机 SNR 加高斯噪声。
* 最后乘以 `valid`，保证补零部分仍是 0（否则模型会把长度判断错）。

### 4.10 `code/utils/optim.py` —— 优化器与学习率

* `build_optimizer`：把参数分成“做权重衰减”和“不做”两组（偏置、LayerNorm、位置编码等不做）。`optimizer="muon"` 时，主干的矩阵参数交给 Muon，其余交给 AdamW。
* `build_scheduler`：前 `warmup_steps` 步学习率线性升高，之后按余弦曲线降到 1%。
* `ModelEMA`：维护一份“影子模型”，每步 `影子 = 0.999×影子 + 0.001×当前`，用于评估和导出，比当前模型更稳。

### 4.11 `code/training/tasks_v2.py` —— 损失与评估

* `ClassificationTaskV2.loss`：增强 → 前向 → 交叉熵（带标签平滑）；调制任务再加调制族交叉熵、可选 SupCon。
* `ParameterTaskV2.loss`：用 codec 编码真值；每个参数单独算损失（只在有效样本上平均），7 个参数取平均；再加 pattern 交叉熵。
* `evaluate`：分类给出 acc 与宏平均召回；参数给出 pattern 准确率、每个参数在物理单位下的 RMSE、范围归一化后的平均 RMSE 和代理精准度。

### 4.12 `code/training/trainer.py` —— 训练循环

`BaseTrainer.fit()` 是所有训练共同的“骨架”：

```
for epoch in 范围:
    values = train_one_epoch()        # 子类实现：训练若干步
    values += on_epoch_end(epoch)     # 子类实现：评估、导出最佳
    写日志 / TensorBoard / 断点文件
    if should_stop(): break           # 早停
结束后画曲线、on_finish()
```

* `MAEPretrainerV2`：每步取一个 batch → `forward_pretrain` → 反向 → 更新；每个周期末在 val 上用固定随机种子算重建损失，变好就保存 `mae_pretrained.pt`，连续 5 次没变好就停止。
* `MultitaskFinetunerV2`：每步
  1. 清空梯度；
  2. 对三个任务依次：取 batch → 算损失 × 权重 → `backward()`（梯度会累加）；
  3. （可选）PCGrad 调整共享参数的梯度；
  4. 梯度裁剪 → 一次 `optimizer.step()` → 学习率调度 → 更新 EMA。
  周期末用 EMA 模型评估三项任务，计算代理总分，创新高就 `export_model_v2` 导出。

### 4.13 入口脚本

* `pretrain_mae.py`：读配置 → 建模型 → 建（分桶的）train/val loader → `MAEPretrainerV2(...).fit()`。
* `finetune_multitask.py`：读配置 → 准备 codec → 写入模型配置 → 建模型 → 校验参数量与共享率 → 加载预训练权重 → `MultitaskFinetunerV2(...).fit()` →（可选）test 评估。
* `training/common.py` 的 `load_mae_pretrained`：**修过一个重要 bug**——旧版永远加载不到预训练权重。现在会打印 “loaded ... N tensors”，N 必须大于 0。

### 4.14 `code/infer.py` —— 推理

1. `load_model`：读 `config.json` → `build_model` → 加载 `model.safetensors` → 挂上 `codec`。
2. `infer_file`：读 Parquet → 把每行 iq 转成数组 → 按 (任务, 长度) 分组 → 每组按 batch 推理（bf16）→ 结果按原始行号放回 → 写 JSONL。
3. 日志里有解析耗时、模型耗时、每样本毫秒数、吞吐。

### 4.15 工具脚本速查

| 想做的事 | 命令 |
|---|---|
| 看显存会不会爆 | `python utils/measure_vram.py --config code/configs/default.json --find_max_bs` |
| 看推理多快 | `python utils/benchmark_latency.py --weights_dir weights` |
| 看哪个算子最慢 | `python utils/profile_model.py --weights_dir weights` |
| 看调制错在哪 | `python code/utils/bad_case_analysis.py --task modulation` |
| 看参数分布 | `python utils/fit_param_codec.py` |
| 导出 ONNX | `python utils/export_onnx.py --weights_dir weights` |

---

## 5. PyTorch 写法速查

| 写法 | 含义 |
|---|---|
| `nn.Module` / `forward` | 网络模块；调用 `model(x)` 时自动执行 `forward` |
| `nn.Parameter` | 需要训练的张量（如 `cls_token`） |
| `register_buffer(..., persistent=False)` | 跟随模型搬到 GPU，但不训练、不存进权重文件（如 DFT 矩阵） |
| `x.view / reshape / permute / transpose` | 改形状 / 调换维度顺序 |
| `torch.cat([a,b], dim=1)` | 沿第 1 维拼接（拼 token 序列） |
| `masked_fill(mask, v)` | mask 为 True 的位置填 v（注意力里填 −inf 表示“忽略”） |
| `torch.gather(x, 1, idx)` | 按下标取值（每个样本取不同位置） |
| `scatter_add_` | 按下标累加（星座直方图） |
| `torch.einsum("blhd,hd->bhl", k, q)` | 爱因斯坦求和，按字母对应维度相乘求和 |
| `F.scaled_dot_product_attention` | PyTorch 内置高效注意力（自动用 FlashAttention） |
| `torch.autocast(..., dtype=torch.bfloat16)` | 混合精度：矩阵乘用 bf16 提速，敏感运算保持 fp32 |
| `loss.backward()` | 反向传播算梯度；多次调用梯度会**累加** |
| `optimizer.zero_grad()` / `step()` | 清空梯度 / 按梯度更新参数 |
| `torch.no_grad()` / `inference_mode()` | 推理时不记录计算图，省显存 |
| `model.train()` / `model.eval()` | 切换训练/评估模式（影响 dropout 等） |
| `state_dict()` / `load_state_dict()` | 导出 / 加载全部权重（字典：名字 → 张量） |
| `safetensors.save_file` | 安全、快速的权重文件格式（赛题要求） |
| `DataLoader(batch_sampler=...)` | 用自定义采样器决定 batch 组成 |

---

## 6. 常见问题

* **pytest 失败**：先看是哪个测试。`test_dft_matmul_matches_fft` 失败说明 torch.fft 环境问题；`test_infer_end_to_end` 失败多半是 pandas/pyarrow 版本问题。把报错贴出来即可。
* **日志显示 `loaded MAE pretrained weights ... 0 tensors`**：预训练配置与微调配置的结构参数（d_model、层数、视角开关）不一致，必须用同一份 model 配置。
* **显存不够**：减小 `batch_size_pretrain/batch_size`，或设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
* **想关闭所有融合算子排查问题**：`EM_FUSED_OPS=0 python ...`。
* **改了数据划分**：微调加 `--refit_codec` 重新拟合参数编码器。
* **训练太慢**：确认 `num_workers` ≥ 8、`length_bucketing: true`；首次运行生成 `lengths.npy` 会慢一些。

---

## 7. 如何自己改代码（三个例子）

1. **加一个新视角（例如循环谱）**：在 `dsp.py` 的 `forward` 里算出 `[B, M]` 特征放进返回字典 → 在 `multiview.py` 的 `__init__` 加 `self.xxx_proj` 与位置编码 → 在 `forward` 的 `parts.append(...)` 处拼进去 → `TYPE_IDS` 加一个类型。
2. **加一种主干层**：在 `layers/conformer.py` 写一个和 `AttnBlock` 同样 `forward(x, pad_mask, time_slice)` 签名的类，加进 `V2_LAYER_TYPES`，配置里就能用这个名字。
3. **改参数损失**：只改 `tasks_v2.py` 的 `ParameterTaskV2.loss`；评估与推理不受影响。
