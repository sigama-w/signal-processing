# 本地运行资源需求与自测报告（模板，请用实测数据填写）

## 1. 硬件与软件
| 项 | 值 |
|---|---|
| GPU | NVIDIA A800 80GB ×1 |
| 驱动 / CUDA | 待填 / 12.x |
| OS | Ubuntu 22.04 |
| Python / PyTorch | 3.10 / 2.3.1 |

## 2. 模型规模
| 项 | 值（`finetune` 日志首行） |
|---|---|
| 总参数 / 部署参数 | 待填 |
| 共享率 | 待填（>80%） |

## 3. 显存（`python utils/measure_vram.py --config code/configs/default.json`）
| 阶段 | batch | max_allocated | max_reserved |
|---|---|---|---|
| 预训练 | 512 | | |
| 微调（三任务） | 256 | | |

## 4. 训练耗时
| 阶段 | 步数 | 墙钟时间 | 样本/秒 |
|---|---|---|---|
| 预训练 | | | |
| 微调 | | | |

## 5. 推理时延（`python utils/benchmark_latency.py --weights_dir weights`）
| 精度 | batch | 长度 | P50 (ms) | P95 (ms) | 每样本 (µs) |
|---|---|---|---|---|---|
| fp32 | 256 | 1024 | | | |
| bf16 | 256 | 1024 | | | |
| bf16 | 256 | 512 | | | |

## 6. 精度（`finetune_multitask.py --eval_test`，test 划分）
| 任务 | 指标 | v1 | v2 |
|---|---|---|---|
| protocol | acc / macro recall | 0.950 / 0.950（val, epoch 4） | |
| modulation | acc / macro recall | 0.626 / 0.626（val, epoch 4） | |
| parameter | pattern acc / 平均 nRMSE | | |

## 7. 推理镜像
`docker run --gpus all --rm -v $PWD/data:/data c01_em:v2 --input /data/test.parquet --output /data/pred.jsonl`
