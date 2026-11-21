# C01 电磁基座模型优化方案（v2）

> 适用代码：本仓库 v2（`model.arch = "v2"`）。v1 代码与配置（`code/configs/default_v1_legacy.json`）完整保留，可随时对照。
> 说明：v2 代码在交付环境中**未能运行 PyTorch**（无 torch），已完成语法检查与 DSP 公式的 numpy 数值校验；
> 请在 A800 上先执行 §12 的验证清单，再开始正式训练。文中“预期收益”为基于原理与文献的判断，需以消融实验为准。

---

## 0. 结论先行：比 10 个问题更紧急的 7 个缺陷

| # | 缺陷 | 影响 | v2 处理 |
|---|---|---|---|
| P0-1 | `load_mae_pretrained` 按 `state["tokenizer"]` 查找，但 `export_pretrained` 保存的是扁平键 `tokenizer.xxx` | **预训练权重从未进入微调**（日志 `loaded ... []` 为空列表），预训练算力全部浪费 | 按前缀筛选扁平键 + `strict=False`，打印实际加载张量数（`training/common.py`） |
| P0-2 | MAE 中可见 token 交叉注意**整段信号**的 FFT（幅度+相位 = 无损变换） | 被掩码内容可从频谱 token 完整恢复，预训练任务被“作弊”，表征学不到东西 | 全局视图一律从“掩码后的信号”计算（`dsp.SignalFrontEnd(sample_keep=...)`） |
| P0-3 | 39.7% 的 512 点样本补零到 1024，**无 padding mask** | 一半 token 是零却参与注意力，污染 CLS 表征；且浪费 ~20% 算力 | 自动推断有效长度 → key padding mask；批内按长度分桶 + 裁剪补零 token |
| P0-4 | 200 epoch × warmup 10% = **20 个 epoch 都在 warmup**；patience=10 | 你日志里 epoch 4 的 lr=7e-5 仍在爬坡；早停会在余弦退火开始前触发 | warmup 按步数（2000 步）；epoch 改为 `steps_per_epoch` 定义的评估周期 |
| P1-1 | 微调中三任务**各自 `optimizer.step()`** | 共享参数被交替拉扯，Adam 动量在三个目标间串扰 | 三任务梯度累加后一次优化步，可选 PCGrad / 不确定性加权 |
| P1-2 | 信号存在大载频偏移且严重过采样（样本图：64QAM 星座糊成圆环） | 不做频偏校正时 QAM 阶数、PSK 族在原始星座/统计上几乎不可分 → 调制 acc 62.6% 的主因之一 | 物理前端：频域质心估 CFO + 整 bin 校正 + 带外抑制；M 次方谱、累积量 |
| P1-3 | 600M~1B 参数网格 | 复赛 30 分按**推理时延排名**；显存逼近/超过 80G（§6）；120 万样本用不满 | 默认 ~35M 参数（d=384, 10 层），网格 256~768 宽 |

另：`tests/test_model_pipeline.py` 引用了不存在的 `parameter_loss`（测试本身是坏的），已修复。

---

## 1. 数据视角与参数尺度（问题 1）

### 1.1 为什么只用 IQ + FFT 不够

从 `figures/real_samples` 可直接读出数据的物理特点：

* **载频偏移大**：64QAM 样本谱峰在 +50 kHz（按 1 MHz 标称采样率），星座呈圆环；
* **严重过采样**：QPSK 占用带宽约为采样带宽的 15~30%，原始采样点星座是“云团”；带外 70%+ 频点全是噪声；
* **幅度跨 5 个数量级**：按峰值归一化后丢失了绝对功率（Idle background 的关键线索）；
* **雷达参数在采样点量级**：time_delay 1~10 µs 对应 3~30 个采样点，PRI/脉冲数体现为包络周期性。

### 1.2 v2 多视角前端 `code/models/dsp.py`（无可学习参数）

| 视角 | 计算 | 主要服务任务 |
|---|---|---|
| 时域 6 通道 | I、Q、校正后 I/Q、|x|、瞬时频率 | 全部 |
| STFT（帧移 = patch，与时域 token 一一对齐） | 32 点 Hann 窗，对数幅度 | 调制（FSK/LFM/CSS）、体制 |
| 全局非线性谱 3 通道 | log|X|、log|FFT(xc²)|、log|FFT(xc⁴)| | **M 次方谱线**区分 BPSK/QPSK/QAM，频偏/带宽 |
| 包络自相关（2N 零填充线性 ACF） | lag 0..511 | **PRI、脉冲数**、码速率 |
| 星座密度图 32×32 | 校正后信号 2D 直方图 | QAM/PSK 阶数 |
| 统计量 16 维 | 原始功率、CFO、占用带宽、SNR、|C20|/|C40|/|C41|/C42/M63、PAPR、占空比、峰度… | 全部（经典 AMC 特征，Swami & Sadler 2000） |

工程要点：
* **DFT 全部用实数矩阵乘实现**（不依赖 `torch.fft` 和复数 dtype）：昇腾 Cube 单元最擅长 GEMM，复数/FFT 算子支持不完整；导出 ONNX/ATC 时没有 DFT 算子兼容问题。1024 点 DFT 每样本约 2M 次乘加，远小于主干。
* 噪底用“最低 50% 频点均值”（`topk`），不用 `median`（ONNX 无对应算子）。
* 前端强制 fp32（autocast 关闭）。
* 归一化由“按峰值”改为“按有效长度的单位功率（RMS）”，抗尖峰；绝对功率以 log 形式进入统计 token。

已用 numpy 校验：实数 DFT 与 `np.fft` 误差 ~1e-11；QPSK + 0.0625 归一化频偏时 CFO 估计 0.0630，校正后 x⁴ 谱线回到 DC（`tests/test_dsp_numpy.py`）。

### 1.3 7 个参数：数量不一致、尺度不均、量纲不同

**现状问题**：min-max → Sigmoid → SmoothL1，逐元素平均。
* symbol_rate p25=100、中位 300、p75=1000、max 2000：min-max 后多数样本挤在 0~0.15，梯度由少数大值主导；
* number_of_pulses 是 2~6 的整数，回归会输出 3.6 这类非法值；
* Sigmoid 边界饱和；雷达 4 参数天然比通信 3 参数权重大。

**v2 `ParamCodec`（`code/utils/param_codec.py`）**：按 **train 划分**统计自动为每个参数选择编码，存 `weights/param_codec.json`：

| 模式 | 判据 | 训练 | 解码 |
|---|---|---|---|
| `cls` | 取值个数 ≤ 32（如 number_of_pulses、档位化码速率/频偏） | 交叉熵 | 后验期望 Σ p_k v_k（**MSE 最优**，与官方 RMSE 对齐）或 argmax |
| `log_zscore` | 正值且 max/min ≥ 20 的长尾连续量 | log 后标准化，Huber | exp 反变换 |
| `zscore` | 其余（pulse_width/pri/time_delay 均匀连续） | 标准化，Huber | 线性反变换 |

损失：**每个参数内部按有效样本平均 → 7 个参数等权平均**（不受 3/4 个数影响），再加 pattern（通信/雷达）辅助 CE。
输出：反变换后裁剪到训练集 [min, max]，再按预测 pattern 把不适用参数置 0（官方要求）。

> 先跑 `python utils/fit_param_codec.py` 看每个参数的唯一值个数与 Top 取值，再决定是否 `--force` 覆盖自动判断。
> 关于官方“固定参考尺度”：任务书未给出具体值与聚合方式。v2 的选择（标准化空间等权 + 原始单位输出）对“逐参数 RMSE/尺度再平均”和“整体 RMSE”两种口径都稳健；若赛方后续公布细则，只需调整 `ref_scale_v2` 与参数损失权重。

---

## 2. layer_types 选型（问题 2）

| 候选 | 原理与适用场景 | 在本任务中的问题 | 结论 |
|---|---|---|---|
| full attention | 全局精确交互 | 序列仅 130~200 token，O(L²) 很便宜 | **保留**（改用 SDPA/FlashAttention） |
| MLA（DeepSeek-V2） | 低秩压缩 KV，收益在**自回归解码的 KV-cache** | 编码器分类无 KV-cache；低秩瓶颈削弱表达；多 2 次矩阵乘 | 不推荐 |
| Gated DeltaNet / Mamba | 线性复杂度长序列建模 | 序列短，优势用不上；仓库实现为 python 逐步循环，训练/推理慢一个数量级，NPU 不友好 | 不推荐 |
| conv1d（深度卷积） | 局部结构、平移等变、极快 | 无全局交互 | 与注意力**组合** |
| **Conformer**（注意力 + 深度卷积，Macaron FFN） | 语音领域证实“卷积捕局部 + 注意力捕全局”优于纯 Transformer（Gulati et al. 2020）；IQ 与语音同属一维振荡信号 | — | **v2 默认** |

电磁领域旁证：SpectrumFM 采用 CNN + MHSA 混合编码器（Zhou et al. 2025）；EMind 也强调 EM 信号“时间稀疏性与连续性并存”。

v2 层类型（`code/models/layers/conformer.py`）：`attn`、`conformer`、`conv`；默认
`["conv", "conformer"×8, "attn"]`（浅层便宜卷积、中层 Conformer、末层纯注意力汇聚）。卷积模块**只作用于时域对齐 token 段**（频谱/星座 token 不是时间序列），padding 位置卷积前清零，512/1024 点数值一致。用 LayerNorm 替代原论文 BatchNorm（对 padding、小 batch 推理与 NPU 更稳）。

消融：`ablation/B1_all_attn.json`、`ablation/B2_all_conv.json`。

---

## 3. 跨域融合 crossdomain_fusion（问题 3）

v1：时域 query 对频谱 K/V 做一次交叉注意力后门控残差，之后频谱信息不再参与；且存在 P0-2 泄漏；频谱相位（/π）几乎是噪声。

v2 三级融合（`code/models/multiview.py`），一一对应任务书“通道、空间或注意力层面的动态对齐与融合”：

1. **通道级**：时域 patch token 与同一时刻 STFT 帧 token 做 SE 式门控 `t' = LN(t + σ(W[t;s]) ⊙ s)`；门控偏置初始化为 −1（先信任时域，训练中逐步打开时频支路）。
2. **空间级**：3 个非线性谱在同一频点按通道堆叠为一个 token（频点对齐）；时间/频率/延迟/星座 token 各有独立位置编码 + 模态类型编码。
3. **注意力级**：`[CLS][STAT][TIME…][SPEC×32][ACF×16][CONST×16]` 单流进入主干，**每一层**自注意力都在做跨域交互（时域 token 可直接查询 x⁴ 谱线、星座簇，反之亦然）。

每个视角可独立开关（`use_stft/use_spec/use_acf/use_const/use_stats`），`ablation/A0~A5` 直接产出技术报告的消融表。

---

## 4. 损失函数（问题 4）

| 阶段 | v1 | v2 | 理由 |
|---|---|---|---|
| 预训练 | MAE，随机单 patch 掩码，MSE 重建 IQ | **SimMIM 块掩码**（4 token=32 采样点/块，掩码率 0.5）；被掩码 token 替换为 mask token 后全序列编码；目标 = 归一化 IQ patch + 对应 STFT 帧（权重 0.5） | 过采样信号单 token 掩码可由邻点插值“作弊”；卷积层需要连续序列；时频双目标对应双流结构 |
| 分类 | CE | CE + label smoothing（protocol 0.05 / modulation 0.1）+ **调制族辅助 CE**（PSK/QAM/FSK/模拟/雷达，权重 0.2） | 层次化监督先学“族”再学“阶”；平滑缓解相近类过度自信 |
| SupCon | 无 | 可选（`supcon_weight.modulation`，建议 0.05~0.1，默认关） | SupCon（Khosla et al. 2020）对相近类边界有帮助，但需大 batch、收益不确定 → 作为消融 `C3_supcon` |
| 回归 | min-max + SmoothL1 | 标准化空间 Huber(δ=1)（可选 MSE）+ 离散参数 CE | 见 §1.3 |

---

## 5. 调制识别 bad case（问题 5）

工具：`python code/utils/bad_case_analysis.py --task modulation --split val`。输出混淆矩阵、Top 混淆对、**准确率随估计 SNR 分桶**、512 vs 1024 点准确率、置信度校准、每个混淆对的 4 联图（时域 / 频偏校正后星座 / 频谱 + x⁴ 谱 / STFT）、自动摘要 `summary.md`。

基于样本图与 AMC 领域经验，预期主要混淆组与对策：

| 混淆组 | 物理原因 | v2 对策 |
|---|---|---|
| 16/32/64/128/256QAM | 未校正频偏 → 星座旋转成环；过采样 + 低 SNR 下高阶 QAM 本就难分 | CFO 校正 + 带外抑制（等效提升 SNR 数 dB）、C42/M63、星座密度 token |
| QPSK / OQPSK / AQPSK / π/4-DQPSK / 8PSK | 差别在相位跳变规律与 M 次方谱线 | x²/x⁴ 谱、瞬时频率通道、调制族辅助头 |
| 2FSK / 4FSK / MSK / GFSK | 频率轨迹差异 | 瞬时频率通道 + 帧对齐 STFT |
| Barker / Polyphase Barker / Frank | 脉内相位编码 | 时域细粒度卷积 stem、瞬时频率 |
| 低 SNR 全类 | 可分性天花板 | 噪声增强、EMA；必要时 TTA（相位旋转平均，代价是时延） |

判读规则：若 SNR≥10 dB 子集已接近 100% 而低 SNR 子集很差 → 天花板问题；若高 SNR 仍成块混淆 → 特征/模型问题（重点检查 figs 中校正后星座是否可分）。

---

## 6. 显存核查（问题 6）

旧 `gen_configs.vram_estimate_gb` 的问题：只在 d512/27L 一个点校准；**未计入反向峰值**；每层 d_model 相关激活按 20 字节/元素计，而 bf16 autocast 下 LayerNorm 输入输出 fp32、RoPE 拼接副本、dropout 掩码等实际约 30 字节（FA 层）~44 字节（MLA 层）；注意力分数按 8 字节计，实际 softmax fp32 输出 + dropout 掩码 + bf16 副本约 7 字节但另有 bf16 拷贝。

按修正口径重估 `grid.json` 的**微调**峰值（微调全序列 130 token，是两阶段中的最大者）：

| 配置 | 旧估算 | 修正峰值（含反向 +25% 与 3GB workspace） |
|---|---|---|
| 1024d_33l（shared85/90） | 62.7 / 62.2 | **74.6 / 74.0** |
| 1024d_36l（85/90/95） | 67.9 / 67.3 / 66.8 | **81.0 / 80.4 / 79.8** |
| 1024d_39l（85/90/95） | 73.1 / 72.5 / 71.9 | **87.3 / 86.7 / 86.1** |
| 1536d_18l（85/90/95） | 57.0 / 56.3 / 55.8 | 66.7 / 66.0 / 65.4 |
| 1536d_21l（85/90/95） | 65.5 / 64.7 / 64.0 | **76.8 / 76.0 / 75.3** |
| 1536d_24l（90/95） | 73.1 / 72.3 | **86.1 / 85.3** |

结论：16 个配置中 1024d_36l、1024d_39l、1536d_24l 共 8 个会 OOM 或贴边，1024d_33l、1536d_21l 也只剩 3~5 GB（allocator 碎片即可打爆）。另外 shared85 系列把任务头做到 4672 宽×2 层（约 1 亿参数）只是为了“凑共享率”，毫无收益。

**行动**：
* 不要再用估算，直接实测：`python utils/measure_vram.py --grid code/configs/grid.json --limit 4`（合成数据，几分钟出结果，报告 pretrain/finetune 两阶段 allocated/reserved，写 `logs/vram/report.json`）；`--find_max_bs` 自动找最大 batch。
* v2 默认 ~35M 参数，batch 256（微调）/512（预训练）预计远低于 20 GB，可用 `measure_vram.py --config code/configs/default.json --find_max_bs` 确认。
* 如仍需大模型：设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少碎片。

---

## 7. CUDA / CANN 融合算子与实测（问题 7）

原则：**先 profiling，再决定写不写算子**。对一个 35M 的编码器，大部分时间在 GEMM 与注意力，手写逐元素算子收益有限；最大的时延收益来自“模型变小 + 分桶裁剪 + 低精度 + 高效注意力内核”。

v2 已提供：

| 层次 | 内容 | 文件 |
|---|---|---|
| 注意力内核 | CUDA：`F.scaled_dot_product_attention`（FlashAttention-2/mem-efficient）；昇腾：`torch_npu.npu_fusion_attention` | `models/ops/fused.py` |
| 逐元素融合 | SwiGLU：CUDA 手写 Triton kernel（前向+反向）；昇腾 `torch_npu.npu_swiglu` | 同上 |
| 可验证迁移 | 每个融合算子首次调用与参考实现 `allclose` 自检，不通过自动回退并记录原因；`EM_FUSED_OPS=0` 一键关闭 | 同上 |
| 编译融合 | `--compile`：对主干 `torch.compile`（Inductor 自动生成 Triton 融合 LN/GELU/残差） | `infer.py` |
| 推理流水 | 按 (task, 长度) 分组成批、512 点只跑 64 个时域 token、bf16、预热、去除 iterrows | `infer.py` |
| 去 CUDA 依赖 | 前端 DFT = 矩阵乘、噪底 = topk、无复数 dtype → ONNX → TensorRT / 昇腾 ATC | `utils/export_onnx.py` |
| 测量工具 | 时延 P50/P95/吞吐（精度 × batch × 长度 × compile） | `utils/benchmark_latency.py` |
| 热点定位 | torch.profiler / torch_npu.profiler，Top-30 算子表 + Chrome trace | `utils/profile_model.py` |

建议实测流程：
```bash
python utils/benchmark_latency.py --config code/configs/default.json \
       --compare code/configs/default_v1_legacy.json --precisions fp32 bf16
python utils/benchmark_latency.py --weights_dir weights --compile
python utils/profile_model.py --weights_dir weights --batch 256
```
若 profile 显示逐元素算子占比 >20%，再考虑把“LayerNorm + 门控融合”或“前端 DFT + 对数幅度”写成专门 kernel；否则优先考虑 INT8/FP8 量化（TensorRT/ATC）。

---

## 8. 借鉴基础模型：后羿大模型（问题 8）

* **后羿**（《现代雷达》2025）：Transformer 架构，面向雷达**数据处理层**（虚警抑制、点迹关联、滤波），基于仿真+实测数据走“预训练-微调”范式。输入是点迹/航迹而非 IQ，因此**架构不宜照搬**；可借鉴的是：多体制多场景统一预训练、仿真数据扩充（可用 `code/utils/data_gen.py` 思路针对弱类生成补充数据，但需在报告中说明来源）、任务头轻量微调的工程链路。
* **EMind**（中电科人工智能研究院等，2025）：IQ 级电磁基座模型，长度自适应多信号打包 + 逐样本掩码。v2 的“按长度分桶 + 有效长度掩码 + 逐样本块掩码”与其思路一致，适合在技术报告中作为对标与引用。
* **SpectrumFM**：CNN + MHSA 编码器、掩码重建预训练，为 Conformer 选型提供电磁领域证据。

---

## 9. 优化器：是否换 Muon（问题 9）

Muon（Keller Jordan 2024；Moonshot 2025）对 2D 权重的动量做 Newton-Schulz 正交化。收益主要在大规模 LLM 预训练的样本效率；对 30M 级编码器微调，收益通常有限。

v2 实现（`code/utils/optim.py`）：主干/stem 中的矩阵用 Muon，嵌入/范数/偏置/任务头用 AdamW；按 Moonshot 做法把更新 RMS 对齐 AdamW（×0.2·√max(m,n)），**可直接沿用 AdamW 学习率**；只有一份动量，省 4B/参数显存；纯 matmul，NPU 友好。

建议：默认 AdamW（betas 0.9/0.95，范数/偏置/嵌入不衰减）建立基线；`ablation/D1_muon.json` 先在预训练阶段对比 val 重建损失下降速度。

---

## 10. epoch 与早停（问题 10）

“10 改 5”不是关键，关键是**调度与早停对齐**：

| 项 | v1 | v2 默认 |
|---|---|---|
| epoch 定义 | 遍历最大任务一次（parameter 82.8 万样本 → 每 epoch 数千步×3） | `steps_per_epoch`（微调 1500 / 预训练 2000 步）= 一个评估周期 |
| 总量 | 200 epoch | 微调 50 周期（75k 步），预训练 30 周期（60k 步） |
| warmup | 10% × 总步数（=20 epoch） | 2000 步（约 3%） |
| 退火 | 余弦到 1% | 余弦到 1%，**按真实总步数**，早停前学习率已充分下降 |
| 预训练早停 | 训练 loss 10 epoch 不降 | **val 重建损失**（固定掩码种子），相对改善 <0.1% 连续 5 次即停，并随时保存最佳 |
| 微调早停 | proxy 分数 10 次不升 | **EMA 权重**评估，`min_delta=1e-4`，patience 6 |

EMA 让验证曲线明显平滑，因此 patience 可以更小而不会误停。

---

## 11. 多任务适配（任务 2）与训练流程的其他改动

* **任务提示**：CLS 叠加任务嵌入，同一主干按任务“调焦”（主干 100% 共享）。
* **任务专属注意力池化**：每个任务用自己的 query 在全部 token 上读出（调制看星座/谱，参数看 ACF/时域沿），与 CLS 拼接。
* **联合优化**：三任务各一个 batch → 梯度累加 → 一次优化步；`pcgrad: true` 启用 PCGrad；`task_weighting: "uncertainty"` 启用 Kendall 不确定性加权（注意它会降低高损失任务的权重，与“加强调制”目标可能冲突，故默认固定权重 protocol 0.5 / modulation 1.0 / parameter 1.0）。
* **数据增强**（按任务区分，只用不改变标签的变换）：随机相位旋转、加性噪声；参数任务禁用共轭/时移（会改变频偏符号/时延）。
* **共享率**：默认配置约 35M 参数，共享（tokenizer+backbone）约 90%，远高于 80% 红线；入口脚本自动校验。

---

## 12. 落地验证清单与实验计划

**第 0 天（必做，约 30 分钟）**
```bash
pip install -r environment/requirements.txt
python -m pytest tests -q                                  # v1 回归 + v2 全链路 + DSP 校验
python utils/measure_vram.py --config code/configs/default.json --find_max_bs
python utils/fit_param_codec.py --data_dir datasets         # 看清 7 个参数的分布与编码模式
python code/training/pretrain_mae.py --smoke_steps 2
python code/training/finetune_multitask.py --smoke_steps 2
python code/infer.py --input datasets/01/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet --output out/pred.jsonl
```

**实验顺序（每步只改一个因素）**
1. E0：v2 默认配置，**不预训练**直接微调（确认 v2 基线 vs v1 62.6%）。
2. E1：v2 预训练 + 微调（确认日志中 `loaded MAE pretrained weights ... N tensors` 且 N>0）。
3. E2：视角消融 A0~A5；E3：主干消融 B1/B2；E4：优化消融 C1~C4、D1~D3。
4. E5：规模扫描 `grid_v2.json`（同时记录 `benchmark_latency.py` 时延，按“精度-时延”帕累托选型）。
5. 每轮跑完 `bad_case_analysis.py`，把 summary.md 与混淆矩阵归档到技术报告。

**技术报告可直接使用的创新点**：物理先验多视角前端（CFO 校正、M 次方谱、包络 ACF、累积量）；三级时-频融合；防泄漏的时频双目标块掩码预训练；任务提示 + 任务专属注意力池化；离散/连续混合参数编解码；“矩阵乘 DFT + 算子自检回退”的国产化迁移方案。

---

## 参考文献

1. Gulati et al., Conformer: Convolution-augmented Transformer for Speech Recognition, Interspeech 2020.
2. Zhou et al., SpectrumFM: A Foundation Model for Intelligent Spectrum Management, arXiv:2505.06256, 2025.
3. Luo et al., EMind: A Foundation Model for Multi-task Electromagnetic Signals Understanding, arXiv:2508.18785, 2025.
4. 叶宇 等, 后羿大模型：基于预训练-微调范式的智能雷达数据处理方法, 现代雷达, 2025, 47(8).
5. He et al., Masked Autoencoders Are Scalable Vision Learners, CVPR 2022；Xie et al., SimMIM, CVPR 2022.
6. Swami & Sadler, Hierarchical Digital Modulation Classification Using Cumulants, IEEE TCOM 2000.
7. Xiao et al., Early Convolutions Help Transformers See Better, NeurIPS 2021.
8. Khosla et al., Supervised Contrastive Learning, NeurIPS 2020.
9. Kendall et al., Multi-Task Learning Using Uncertainty to Weigh Losses, CVPR 2018；Yu et al., Gradient Surgery for Multi-Task Learning (PCGrad), NeurIPS 2020.
10. Jordan, Muon optimizer, 2024；Liu et al., Muon is Scalable for LLM Training, arXiv:2502.16982, 2025.
11. DeepSeek-AI, DeepSeek-V2 (MLA), 2024；Yang et al., Gated Delta Networks, ICLR 2025；Gu & Dao, Mamba, 2023.
