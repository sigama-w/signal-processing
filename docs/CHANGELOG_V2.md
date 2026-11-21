# v2 新增与修改内容说明

> 本文件逐项说明 v2 相对你上传版本（signal-processing-main_261121）新增/修改了什么、怎么用、配置项含义。
> 设计理由见 `docs/OPTIMIZATION_PLAN.md`，代码逐行讲解见 `docs/CODE_WALKTHROUGH.md`。

## 1. 验证状态（请先读）

| 项目 | 状态 |
|---|---|
| 全部 `.py` 语法编译 | ✅ 已通过 |
| DSP 公式（实数 DFT、CFO 估计/校正、STFT 帧对齐）numpy 校验 | ✅ 已通过（`tests/test_dsp_numpy.py`） |
| PyTorch 单元测试 / 冒烟训练 / 推理 | ⚠️ **交付环境无 PyTorch，未运行**。请在 A800 上执行 `python -m pytest tests -q` |
| 官方数据训练、精度、时延、显存 | ❌ 未运行（需你的训练机） |

若 pytest 报错，把完整报错贴给我即可快速修。最可能出问题的是：不同 torch 版本的 SDPA 布尔掩码、Triton 自检、`torch_npu` 算子签名——这三处都带自动回退，即使融合算子失败也会退回参考实现，不影响正确性。

## 2. 文件清单

### 2.1 新增

| 文件 | 作用 |
|---|---|
| `code/models/dsp.py` | 物理先验信号前端：有效长度推断、单位功率归一化、实数矩阵 DFT、CFO 估计与校正、带外抑制、STFT、非线性谱、包络 ACF、星座密度、16 维统计量 |
| `code/models/multiview.py` | 多视角 Tokenizer：卷积 stem、时域⊕STFT 门控融合、谱/ACF/星座/统计 token、类型与位置编码、任务提示、补零 token 裁剪 |
| `code/models/layers/conformer.py` | v2 主干：SDPA 多头注意力、Conformer 卷积模块、`attn`/`conformer`/`conv` 三种层、随机深度、`UnifiedEncoder` |
| `code/models/ops/fused.py`、`ops/__init__.py` | 融合算子分发（CUDA SDPA / Triton SwiGLU / 昇腾 npu_fusion_attention、npu_swiglu）+ 数值自检回退 |
| `code/models/heads_v2.py` | 任务专属注意力池化、分类头（含调制族辅助、SupCon 投影）、参数头（pattern + 回归 + 离散分类） |
| `code/models/foundation_v2.py` | `EMFoundationV2` 组装；`encode/forward/predict/forward_pretrain`；共享率核算；部署权重导出 |
| `code/utils/param_codec.py` | 7 参数编解码器（`cls` / `log_zscore` / `zscore` 自动选择） |
| `code/utils/augment.py` | 按任务区分的 GPU 批量增强 |
| `code/utils/optim.py` | AdamW 分组、Muon、组合优化器、warmup+余弦调度、EMA |
| `code/utils/bad_case_analysis.py` | 分类 bad case 分析（混淆矩阵、SNR 分桶、长度、校准、样本 4 联图、摘要） |
| `code/training/tasks_v2.py` | v2 损失与评估 |
| `utils/measure_vram.py` | 显存实测（合成数据，无需数据集；可批量测 grid、找最大 batch） |
| `utils/benchmark_latency.py` | 推理时延 P50/P95/吞吐基准 |
| `utils/profile_model.py` | 算子级 profiling（CUDA / 昇腾） |
| `utils/export_onnx.py` | 导出 ONNX（TensorRT / ATC） |
| `utils/fit_param_codec.py` | 拟合并查看参数编码与分布 |
| `code/configs/default.json` | **v2 默认配置**（旧默认改名为 `default_v1_legacy.json`） |
| `code/configs/ablation/*.json` | 15 个消融配置（A 视角 / B 主干 / C 多任务 / D 优化） |
| `code/configs/grid_v2.json` | v2 规模网格（256~768 宽） |
| `tests/test_v2_pipeline.py`、`tests/test_dsp_numpy.py` | v2 全链路测试、无 torch 的 DSP 校验 |
| `environment/requirements_infer.txt`、`requirements_lock.txt` | 推理镜像最小依赖、锁定清单模板 |
| `docs/OPTIMIZATION_PLAN.md`、`docs/CHANGELOG_V2.md`、`docs/CODE_WALKTHROUGH.md`、`docs/self_test_report.md` | 文档 |

### 2.2 修改

| 文件 | 改动 |
|---|---|
| `code/training/common.py` | **修复预训练权重从未加载的 bug**；`arch` 加入模型键 |
| `code/training/trainer.py` | 优化器/调度器改由 `utils/optim.py` 构建（warmup 按步数）；新增 `MAEPretrainerV2`、`MultitaskFinetunerV2`、`export_model_v2` |
| `code/training/pretrain_mae.py` | v2 分支（长度分桶 loader、val 早停）；`UnlabeledIQ` 修复重复读 memmap |
| `code/training/finetune_multitask.py` | v2 分支（codec 拟合、v2 loader、test 评估）；`--refit_codec` |
| `code/models/__init__.py` | `build_model` / `model_arch` 工厂 |
| `code/utils/dataset.py` | 返回 `raw_params`；长度缓存 `lengths.npy`；`LengthGroupedBatchSampler`；`build_task_dataloaders_v2`；`load_param_arrays` |
| `code/infer.py` | 支持 v1/v2；(task,长度) 分组；bf16/fp16/fp32；`--compile`；计时与融合算子状态日志 |
| `tests/test_model_pipeline.py` | 修复引用不存在函数的问题 |
| `docker/Dockerfile` | CUDA 12.1 + PyTorch 2.3.1 基础镜像，最小依赖，无 VOLUME |
| `datasets/README.txt` | 填写训练数据来源说明（原为空文件） |
| `utils/gen_configs.py` | 文档头加入显存公式低估的警告 |
| `README.md` | 更新为 v2 使用说明 |

v1 的 `foundation.py / tokenizer.py / backbone.py / frequency.py / heads.py / layers/*` **未改动**，`arch` 缺省为 v1，旧权重与旧配置仍可加载。

## 3. 使用方法

```bash
# 0) 环境与自检
pip install -r environment/requirements.txt
python -m pytest tests -q

# 1) 数据（与 v1 相同，只需一次）
python utils/prepare_official.py

# 2) 查看参数分布 / 生成编码器（可选，微调会自动生成）
python utils/fit_param_codec.py --data_dir datasets --out weights/param_codec.json

# 3) 显存实测
python utils/measure_vram.py --config code/configs/default.json --find_max_bs

# 4) 预训练（v2：SimMIM 块掩码 + val 早停）
python code/training/pretrain_mae.py --config code/configs/default.json --data_dir datasets --out_dir weights

# 5) 多任务微调（自动加载 weights/mae_pretrained.pt；结束后 test 无偏评估）
python code/training/finetune_multitask.py --config code/configs/default.json --data_dir datasets --out_dir weights --eval_test

# 6) 分析调制 bad case
python code/utils/bad_case_analysis.py --task modulation --split val

# 7) 推理与时延
python code/infer.py --input <test.parquet> --output out/pred.jsonl --precision bf16
python utils/benchmark_latency.py --weights_dir weights --compile

# 8) 消融（示例）
python code/training/finetune_multitask.py --config code/configs/ablation/A0_time_only.json --out_dir weights_A0
```

微调日志里请确认出现：`loaded MAE pretrained weights from ...: N tensors`（N>0）。

## 4. v2 配置项说明（`code/configs/default.json`）

### 4.1 model

| 键 | 默认 | 含义 |
|---|---|---|
| `arch` | `"v2"` | `v2` 走新模型；缺省/`v1` 走旧模型 |
| `d_model / n_layers / nhead / ffn_hidden` | 384 / 10 / 6 / 1024 | 主干宽度、层数、头数、SwiGLU 隐层 |
| `layer_types` | conv + conformer×8 + attn | 逐层类型，可写短模式串循环铺满 |
| `conv_kernel` | 15 | 深度卷积核（15 token ≈ 120 采样点感受野/层） |
| `dropout / drop_path` | 0.1 / 0.1 | dropout 与随机深度（线性递增） |
| `seq_len / patch_size / stft_win` | 1024 / 8 / 32 | 输入长度、每 token 采样点数、STFT 窗长 |
| `use_stft/use_spec/use_acf/use_const/use_stats` | 全 true | 各视角开关（消融用） |
| `task_prompt` | true | CLS 叠加任务嵌入 |
| `trim_padding` | true | 批内裁掉全为补零的时域 token |
| `dft_impl` | `"matmul"` | `matmul`（国产化/ONNX 友好）或 `fft`（对照） |
| `head_config` | 见文件 | 各任务头宽度/层数 |
| `mod_family_aux` / `supcon_dim` | true / 128 | 调制族辅助头 / SupCon 投影维（0 关闭） |
| `mask_ratio / mask_block / stft_loss_weight` | 0.5 / 4 / 0.5 | 预训练掩码率、块大小（token）、STFT 重建权重 |
| `param_head_spec / param_codec` | 自动写入 | 由微调脚本根据 codec 写入导出的 config.json，推理自包含 |

### 4.2 training

| 键 | 默认 | 含义 |
|---|---|---|
| `batch_size_pretrain / batch_size(_finetune) / eval_batch_size` | 512 / 256 / 1024 | 各阶段 batch（微调为“每任务”batch） |
| `lr_pretrain / lr_finetune` | 6e-4 / 3e-4 | 各阶段峰值学习率 |
| `optimizer` | `adamw` | 或 `muon` |
| `warmup_steps / min_lr_ratio` | 2000 / 0.01 | warmup 步数、余弦终点比例 |
| `epochs_pretrain / steps_per_epoch_pretrain` | 30 / 2000 | 预训练周期数与每周期步数 |
| `patience_pretrain / min_delta_pretrain / val_batches_pretrain` | 5 / 0.001 / 40 | 预训练 val 早停 |
| `epochs_finetune / steps_per_epoch` | 50 / 1500 | 微调周期数与每周期步数（0=遍历最大任务一次） |
| `patience / min_delta` | 6 / 1e-4 | 微调早停 |
| `ema_decay` | 0.999 | 0 关闭 EMA |
| `length_bucketing` | true | 512/1024 分桶组 batch（单卡生效） |
| `augment` / `augment_policy` | true / 默认策略 | 数据增强开关与自定义策略 |
| `task_weighting / task_weights / pcgrad` | fixed / 0.5,1,1 / false | 多任务加权与 PCGrad |
| `label_smoothing / family_loss_weight` | 0.05,0.1 / 0.2 | 分类正则 |
| `supcon_weight / supcon_temperature` | 0 / 0.1 | SupCon |
| `param_loss / pattern_loss_weight` | huber / 0.3 | 参数损失 |
| `ref_scale_v2` | 0.1 | 本地参数代理精准度的参考尺度（范围归一化 RMSE） |
| `score_weights` | 0.3/0.3/0.4 | 代理总分权重（初赛口径） |
| `codec` | max_classes 32, decode expectation | 参数编码自动判别阈值与解码方式（可加 `force`） |

## 5. 产物（weights/）

| 文件 | 说明 |
|---|---|
| `mae_pretrained.pt` | 预训练最佳共享权重（扁平 state_dict） |
| `param_codec.json` | 参数编解码器 |
| `config.json` | 完整配置（内嵌 `param_codec` 与 `param_head_spec`，推理自包含） |
| `model.safetensors` | 最佳（EMA）部署权重，不含预训练重建头 |
| `finetune_resume.pt / mae_resume.pt` | 断点 |

## 6. 兼容性与迁移

* v1 权重**不能**加载到 v2（结构不同）；v2 需重新预训练 + 微调。
* 想继续用 v1：`--config code/configs/default_v1_legacy.json`（v1 也自动获得预训练加载修复与新调度器；legacy 配置里显式写了 warmup_ratio=0.1，行为不变，建议改为 0.03 或改用 warmup_steps）。
* 多卡（torchrun）仍支持；多卡时长度分桶自动关闭（使用 DistributedSampler）。
