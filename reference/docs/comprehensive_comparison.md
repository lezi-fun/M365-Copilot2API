# 综合对比：我们的方案 vs DS V4 vs Ornith vs Mythos

## 对比维度

| 维度 | 我们自己 | DeepSeek-V4 | Ornith-1.0 | Claude Mythos |
|------|---------|------------|------------|---------------|
| **权重精度** | 3T1B 6-bit (54级) | FP4+FP8 mixed | BF16 (基座) | 未公开 |
| **注意力** | RFF O(nDd) / LUT O(1) | CSA+HCA 混合注意力 | 标准注意力 (基座) | 未公开 |
| **FFN** | HierLUT O(1) / GBMLH | DeepSeekMoE (384/256 experts) | MoE/标准FFN (基座) | 未公开 |
| **残差连接** | 标准 | **mHC** (Birkhoff约束) | 标准 (基座) | 未公开 |
| **优化器** | 自研AdamW | **Muon** (Newton-Schulz) | 标准 (基座) | 未公开 |
| **推理加速** | 3T1B + RFF + LUT 三合一 | FP4 QAT + sparse attn | 无特别加速 | 未公开 |
| **训练策略** | 中国教育6阶段 | sequence渐进 + Anticipatory Routing | **Self-Scaffolding RL** | 未公开 |
| **数据管线** | 全路径探索+错题本+知识书 | 32T tokens 通用预训练 | 基座预训练 + agentic RL | 未公开 |
| **长上下文** | RFF O(n) 理论上支持 | CSA/HCA 1M tokens | 128K | **200K-1M+** |
| **Agent能力** | 全路径探索驱动 | SFT+GRPO specialist → distill | **Self-Scaffolding SWE 82.4%** | 最强，但封闭 |

---

## 逐项对比：谁更强

### 1. 权重量化

| | 我们 | DS V4 | Ornith | Mythos |
|---|------|-------|--------|--------|
| 精度 | **6-bit (54级, 37.5%)** | FP4+FP8 mixed | BF16 | 未公开 |
| 吞吐 | **25.7x vs FP16** (实测) | 理论~2x (FP4×FP8) | 1x | 未公开 |
| 硬件 | P106 6GB 可跑 | H800 80GB 起 | H100/A100 | 云端 |

**我们的优势**：3T1B 在极低精度下的吞吐提升是实测的 25.7x，FP4 QAT 需要专用硬件支持。你的 3T1B 在低端硬件上碾压。

---

### 2. 注意力机制

| | 我们 | DS V4 | Ornith | Mythos |
|---|------|-------|--------|--------|
| 复杂度 | **O(n)** (RFF) / **O(1)** (LUT) | O(n·m) (CSA压缩) | O(n²) | 未公开 |
| KV cache | 无 (RFF) / 极小 (LUT) | 10% of V3 (压缩+稀疏) | 标准KV cache | 未公开 |
| 精度 | 随机权重r≈0 (需训练) | **已验证的1M context** | 标准精度 | 未知 |

**DS V4 优势**：CSA+HCA 是工程验证过的方案，1M context 实测。你的 RFF 精度需训练后才确认，理论优势大但没验证。

**我们的设计可以借鉴**：DS V4 的 CSA 压缩策略（m=4, m'=128, top-k=512）可以适配到你已有的 RFF 上——压缩后再用 RFF 投影，而不是原始 QK。

---

### 3. 残差连接

| | 我们 | DS V4 | Ornith | Mythos |
|---|------|-------|--------|--------|
| 方案 | 标准残差 | **mHC** (n_hc=4) | 标准 | 未公开 |
| 谱范数 | 可能>1 | 约束≤1 | 无约束 | 未公开 |
| 可借鉴性 | - | 可以直接用 | - | - |

**mHC 推荐直接采用**：不过是加几行代码的事（Sinkhorn-Knopp 迭代约束 B_l 到双随机矩阵），能稳定信号传播。适合集成到链式反应的每个 node 中。

---

### 4. 优化器

| | 我们 | DS V4 | Ornith | Mythos |
|---|------|-------|--------|--------|
| 方案 | 自研AdamW | **Muon** | AdamW (基座) | 未公开 |
| 收敛 | 标准 | 更快 (实测) | 标准 | 未公开 |
| 实现 | easy | Newton-Schulz正交化 | - | - |

**Muon 推荐采用**：核心是每步对梯度矩阵做正交化（Newton-Schulz 迭代），对 MoE/大矩阵有效。代码量不大，可以直接实现替换 AdamW。

**Anticipatory Routing** 也值得抄：第 t 步的路由用 θ_{t-δ} 的旧权重，减少 loss spike。

---

### 5. 训练策略

| | 我们 | DS V4 | Ornith | Mythos |
|---|------|-------|--------|--------|
| 预训练 | 无（从基座开始） | 32T tokens 从头训 | 基座预训练 | 未公开 |
| SFT | **中国教育6阶段** | standard SFT | - | 未公开 |
| RL | 无（分类loss替代） | GRPO specialist | **Self-Scaffolding GRPO** | RLHF |
| post-training | 无 | **On-Policy Distillation** | pipeline-RL | 未公开 |
| 稳定性 | 未验证 | Anticipatory Routing | staleness weight | 未公开 |

**我们 vs Ornith**：你们的 Self-Scaffolding 和我们的全路径探索+教育框架是互补关系。Ornith 的 scaffold ≈ 你数据管线的"探索策略"，但 Ornith 用 RL 学，你用监督学习。可以结合：在清华北大的"自主选课"阶段引入 Self-Scaffolding 的 RL 思路。

**DS V4 的 On-Policy Distillation**：训练多个领域 specialist → 统一蒸馏成单一模型。后期可以借鉴。

---

### 6. 数据管线

| | 我们 | DS V4 | Ornith | Mythos |
|---|------|-------|--------|--------|
| 预训练数据 | 无 | 32T tokens 通用+代码 | 基座预训练 | 未公开 |
| 训练数据 | **全路径探索+错题本+知识书** | standard SFT data | agentic rollout | 未公开 |
| 创新 | 错误反思+规则提取 | 无特别 | 自生成scaffold | 未公开 |

**我们的独特优势**：全路径探索、错题本、知识书、IF-THEN-ELSE 规则提取——这些是 DS V4 和 Ornith 都没有的。这是核心竞争力。

---

### 7. 长上下文

| | 我们 | DS V4 | Ornith | Mythos |
|---|------|-------|--------|--------|
| 能力 | 理论支持 (RFF O(n)) | **实测 1M tokens** | 128K | **1M tokens** |
| 效率 | 未实测 | 27% FLOPs of V3 | 无特别 | 未知 |

DS V4 和 Mythos 都是实测 1M context 的。我们只有理论。

---

### 8. Agent能力

| | 我们 | DS V4 | Ornith | Mythos |
|---|------|-------|--------|--------|
| SWE-Bench | 未测 | 80.6 | **82.4** | 未公开 (Mythos Preview 49.9%?) |
| 自我改进 | **全路径探索** | GRPO specialist | **Self-Scaffolding** | recursive self-correction loops |
| 工具使用 | 数据管线已有 | 有 | 有 | 原生系统工具集成 |
| 安全 | 没有 | 没有特别 | 防reward hacking 3层 | **3层沙箱+classifiers** |

**Ornith 在 agentic coding 上最强**（82.4 SWE-Bench），它的 Self-Scaffolding 理念和你的全路径探索高度互补。

**Mythos 最神秘但最强**，但 Anthropic 没公开细节。

---

## 总结：我们可以拿什么

| 可以直接抄的 | 参考思路的 | 我们比他们强的 |
|-------------|-----------|--------------|
| **mHC 残差** (几行代码搞定) | **CSA/HCA 压缩策略** (适配到RFF) | **3T1B 量化** (25.7x, 低硬件可跑) |
| **Muon 优化器** (替换AdamW) | **Self-Scaffolding** (整合到训练后期) | **全路径探索数据管线** |
| **Anticipatory Routing** | **On-Policy Distillation** (多专→统一) | **错题本+知识书** |
| **序列长度渐进** 4K→16K→64K→1M | FP4 QAT 思路 (你做的是6-bit, 更激进) | **中国教育训练框架** |
| Sqrt(Softplus) 路由激活函数 | | **链式反应动态深度** |

**最大缺口**：我们的所有"优势"都是理论/设计阶段，DS V4 和 Mythos 是已经跑通验证的工程。差距是"有没有跑通"而不是"谁的设计更好"。

你觉得这个对比合理吗？要接着做什么？
