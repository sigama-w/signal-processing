# C01 多类型电磁信号感知 — 物理先验多视角电磁基座模型（v2）

单一共享基座同时完成 **信号体制识别（8 类）/ 调制识别（23 类）/ 7 个物理参数估计**。
v2 在 v1 基础上重构了输入表征、主干、多任务优化与推理链路，并修复了若干关键缺陷（见下）。

> ⚠️ 交付环境无 PyTorch：v2 代码已通过语法检查与 DSP 公式 numpy 校验，但**尚未在 GPU 上运行**。
> 请先执行 `python -m pytest tests -q` 与两阶段 `--smoke_steps 2` 冒烟，再正式训练。

## v2 要点

| 模块 | v2 方案 |
|---|---|
| 输入（任务 1） | 物理前端：有效长度掩码、单位功率归一化、**载频偏移估计与校正**、带外抑制、STFT、x²/x⁴ 非线性谱、包络自相关、星座密度、16 维统计量（实数矩阵 DFT，NPU/ONNX 友好） |
| 融合（任务 1） | 通道级（时域⊕对齐 STFT 门控）+ 空间级（频点对齐堆叠、模态/位置编码）+ 注意力级（单流联合序列） |
| 主干 | Conformer 式 `conv / conformer / attn` 混合层，SDPA/FlashAttention，默认 ~35M 参数 |
| 多任务（任务 2） | 任务提示 + 任务专属注意力池化；三任务梯度累加单步更新；可选 PCGrad / 不确定性加权；EMA |
| 参数估计 | `ParamCodec` 自动为 7 个参数选择 分类/对数标准化/标准化 编码，参数等权损失 |
| 预训练 | SimMIM 块掩码 + 时域/STFT 双目标 + 防泄漏全局视图 + val 早停 |
| 效率（任务 3） | 长度分桶 + 补零 token 裁剪、bf16、融合算子自检回退（CUDA Triton / 昇腾 torch_npu）、torch.compile、ONNX 导出、时延/显存/profiling 工具 |

**修复的关键缺陷**：预训练权重从未加载进微调；MAE 通过整段 FFT 泄漏被掩码信息；512 点样本补零无掩码；warmup 占 20 个 epoch；三任务各自 step。

## 快速开始

```bash
pip install -r environment/requirements.txt
python -m pytest tests -q
python utils/prepare_official.py                       # 官方数据 -> memmap（只需一次）
python utils/measure_vram.py --config code/configs/default.json --find_max_bs
python code/training/pretrain_mae.py --data_dir datasets --out_dir weights
python code/training/finetune_multitask.py --data_dir datasets --out_dir weights --eval_test
python code/utils/bad_case_analysis.py --task modulation --split val
python code/infer.py --input datasets/01/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet \
                     --output out/test_predictions.jsonl
python utils/benchmark_latency.py --weights_dir weights
```

v1 配置仍可用：`--config code/configs/default_v1_legacy.json`。

## 文档

- [优化方案（10 个问题的分析与决策）](docs/OPTIMIZATION_PLAN.md)
- [v2 新增与修改内容说明](docs/CHANGELOG_V2.md)
- [代码解读报告](docs/CODE_WALKTHROUGH.md)
- [本地资源与自测报告模板](docs/self_test_report.md)
- 历史文档：[v1 审查](docs/CHANGELOG_AND_REVIEW.md)、[训练指导](docs/training_guide.md)、[数据分析](docs/data_analysis_report.md)、[显存分析(v1)](docs/gpu_memory_analysis.md)

## 赛题约束

总参数 < 1B（v2 默认约 35M）；共享权重 > 80%（v2 约 90%，入口自动校验）；单一基座、无集成、无外挂知识库。
