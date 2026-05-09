# 项目交接文档 — Distributional Risk-Aware Robust Safe RL

> 用途：跨窗口交接的完整项目背景。新窗口起手时附上此文档 + 一句"我现在想做 X"。
> 创建时间：2026-05-08
> 上一份交接：`PRISM_handoff_memo_.md`（2026-05-08 早些时候，已 outdated）

---

## 0. 一句话项目状态

**当前共识方向**：放弃 PRISM 主算法（已被实验数据否决），转向 **D-EVO-Choquet**（Distributional EVO with Choquet Aggregation），此算法需要你重新取一个合适的名字，主战场迁移到 **Robust-Gymnasium** 的 RobustSafety MuJoCo 系列。最终目标：**1 篇 paper + 1 个专利 → 毕业**。

**毕业论文 title 已定**：风险感知的值分布鲁棒安全强化学习。

**当前 next step**：先做 5 GPU-h 的 PoC 验证（接口 + GPD 拟合可行性），再决定是否全量投入。

---

## 1. 我们在干什么 / 想干什么

### 1.1 长期目标

完成毕业三件套：
1. **1 篇会议或期刊 paper**（投 ccf-b及以上）
2. **1 个发明专利**（技术方案与 paper 解耦，专利写完整算法即可）
3. **毕业论文**（中文学位论文，6 章常规结构）

### 1.2 当前选定的技术方向：**D-EVO-Choquet**

**核心想法**：在 EVO（ICML 2025）的基础上做 distributional 升级 + multi-source 升级 + robust 扩展。

**三个技术贡献**（对应论文 title 三个关键词）：

1. **风险感知** = **多源 EVT-CVaR**：每个 risk source 单独维护 GPD head，从 distributional critic 的尾部 quantile 拟合
2. **值分布** = **Distributional cost critic**：替换 EVO 的标量 V_C(s)，用 QR-style quantile critic 输出 N=51 个 quantile
3. **鲁棒安全** = **Choquet 多源聚合 + robust evaluation**：跨 risk source 的尾部估计用 Choquet integral 聚合（复用 fuz B1 fix 后的代码），允许非加性交互；在 robust-gymnasium 的扰动下评估

### 1.3 与已有工作的 Differentiation Table

| 方法 | distributional cost | EVT tail | robust to perturbation | aggregation |
|---|---|---|---|---|
| CPO/PPO-Lag | ✗ | ✗ | ✗ | — |
| EVAC (NeurIPS 2023) | ✓ (no constraint) | ✓ | ✗ | — |
| EVO (ICML 2025) | ✗ (scalar V_C + GPD on C) | ✓ (single GPD) | ✗ | — |
| FuzRL (NeurIPS 2025) | ✗ | ✗ | ✓ (eval only) | Choquet (有 B1 bug) |
| MICE (ICML 2025) | ✗ | ✗ (count-based) | ✗ | — |
| **D-EVO-Choquet (ours)** | **✓** | **✓ (multi-source GPD)** | **✓** | **Choquet (sigmoid+reg)** |

**每行都有 baseline cover 一部分，没有 baseline cover 全部**——这是 paper intro 的标准 differentiation 模板。

### 1.4 演变历程（重要 context）

**初衷（已废弃）**：发表 PRISM 论文（fuz + EVT-CVaR + 5 个 OT shells），证明它在 safe RL 任务上 dominate baseline。

**为什么放弃 PRISM**：3 seed 时 PRISM 看起来好，10 seed 后被证明是 cherry-pick。具体见 §2 实验结果。OT 在 cartpole 上系统性有害（4/4 对照都让 metric 变差）。

**关于 fuz B1 fix 的处理决策**：fuz 原版用 softmax 导致 Choquet 退化为加权平均，作者用 sigmoid+reg 修复后效果显著提升（34.0 → 64.4）。**决定不把 B1 fix 当作论文 contribution，而是作为 implementation detail（appendix 一句话）**——理由：
- 单 B1 fix 不够发好 paper 也不够发专利
- 在 D-EVO-Choquet 中，baseline 用 sigmoid 实现的 Fuz-PPO-Lag 是 fair comparison
- 这种处理在 ML 论文里是 standard practice（baseline 用 best implementation）

---

## 2. 当前进度与实验结果

### 2.1 代码进度

完成度（基于 `evt_otp_fuz_v5_ablation.tar.gz`）：

| 模块 | 状态 |
|---|---|
| PPO-Lagrangian 框架 | ✅ 跑通 |
| Fuz (Choquet, sigmoid + reg, B1 fixed) | ✅ 跑通 |
| EVT-CVaR module (GPD MLE) | ✅ 跑通（但效果待商榷，见 §2.4） |
| OT-shell perturbation | ✅ 跑通（但被证明有害） |
| 8-cell 模块化 ablation 接口 | ✅ 完成（`use_fuzzy / use_evt / n_ot_shells` 3 个正交 CLI flag） |
| `evaluate_scg.py` 三通道分列输出 | ✅ 完成（FuzRL Table 1 风格） |
| `plot_results.py` 阴影带 + ablation tag 兼容 | ✅ 完成 |

### 2.2 已有实验数据：cartpole_stab × 10 seed × paper_table3 协议

**完整 9 method × 3 channel 表（image 1，AvgRet ± std / AvgRisk ± std，10 seed）**：

| Method | AvgRet(Obs) | AvgRisk(Obs) | AvgRet(Act) | AvgRisk(Act) | AvgRet(Dyn) | AvgRisk(Dyn) |
|---|---|---|---|---|---|---|
| baseline | 22.4±35.4 | 0.697±0.229 | 26.8±46.6 | 0.693±0.237 | 27.2±47.6 | 0.693±0.236 |
| f0_e1_o0 (EVT only) | 16.8±26.1 | 0.677±0.228 | 20.9±35.4 | 0.677±0.228 | 21.6±36.9 | 0.676±0.227 |
| f0_e1_o5 (EVT+OT) | 4.3±5.1 | 0.599±0.114 | 5.0±6.8 | 0.596±0.117 | 5.2±6.9 | 0.596±0.119 |
| f1_e0_o5 (fuz+OT) | 31.1±32.7 | 0.410±0.177 | 34.8±42.8 | 0.415±0.180 | 35.5±44.0 | 0.415±0.179 |
| **f1_e1_o0 (fuz+EVT no OT)** | **48.8±28.9** | **0.215±0.267** | **59.5±38.4** | **0.215±0.266** | **61.1±39.4** | **0.213±0.265** |
| **fuz (= f1_e0_o0)** | **61.5±57.0** | **0.269±0.333** | **76.6±72.2** | **0.269±0.338** | **78.3±73.9** | **0.270±0.337** |
| fuz_softmax (FuzRL parity) | 33.0±58.7 | 0.664±0.264 | 37.6±69.2 | 0.664±0.266 | 38.6±71.1 | 0.664±0.266 |
| otp | 1.3±1.1 | 0.633±0.026 | 1.2±1.1 | 0.636±0.026 | 1.1±1.1 | 0.636±0.024 |
| prism (= f1_e1_o5) | 39.9±31.0 | 0.289±0.265 | 53.5±48.0 | 0.287±0.271 | 55.1±49.1 | 0.287±0.271 |

### 2.3 4 个关键发现（cartpole_stab）

**Finding 1: Method ranking 不依赖 perturbation channel** — 每个 method 在 obs/act/dyn 三列内部数值高度一致（fuz: 61.5/76.6/78.3，prism: 39.9/53.5/55.1）。可以放心做 channel-aggregated 分析。

**Finding 2: PRISM 被 f1_e1_o0 严格 Pareto-dominate**

Pareto 前沿只有两个点：
- **fuz** (61.5, 0.269) — 最高 return
- **f1_e1_o0** (48.8, 0.215) — 最低 risk
- **prism** (39.9, 0.289) — **同时被 fuz 和 f1_e1_o0 dominate**（return 更低 + risk 更高）

**Finding 3: OT 系统性有害（4/4 对照都成立）**

| 对照 | 加 OT | 不加 OT | Δ Return | Δ Risk |
|---|---|---|---|---|
| fuz+EVT | prism (39.9 / 0.289) | f1_e1_o0 (48.8 / 0.215) | **+22%** | **−26%** |
| fuz only | f1_e0_o5 (31.1 / 0.410) | fuz (61.5 / 0.269) | +98% | −34% |
| EVT only | f0_e1_o5 (4.3 / 0.599) | f0_e1_o0 (16.8 / 0.677) | +291% | +13% |
| baseline | otp (1.3 / 0.633) | baseline (22.4 / 0.697) | +1623% | −9% |

**Finding 4: cartpole 上没有 safe-attractor pathology** — 所有 9 个 method 的 Lagrangian penalty 都在持续上涨，没人 collapse。fuz 赢的原因不是"装死安全吸引子"，是真的找到了更高 return 策略。

### 2.4 三个值得 push back 的诊断结论

**(a) EVT 在 cartpole 上"中性"**：fuz vs f1_e1_o0 是 64.4 vs 51.0（cross-channel mean），加 EVT 后 mean 反而降了 13.4。这不是 EVT 的失败，是**用错了**——
- **EVAC/EVO 的 EVT 是建模 distributional critic 的尾部 / cumulative cost C 的尾部**
- **PRISM 的 EVT 是建模标量 V_C 在扰动邻域的 K 个值** ← 这没有 distributional grounding，拟合到的是 critic 函数的 spatial smoothness
- 这是为什么必须升级到 distributional cost critic（见 §3.1.2 / §3.1.3 文献）

**(b) cartpole_stab 太简单** — 4 维 state，cost 几乎是 bimodal（要么 violate 要么不 violate），尾部几乎不存在。GPD 在 bimodal 上拟合是 ill-defined。**这是 cartpole_stab 不能作为 main result 的核心原因**。

**(c) on-policy vs off-policy 不是问题** — 用户曾猜测 EVT 失效是因为 EVAC 是 off-policy 而 PRISM 是 on-policy。EVO 已经证明 on-policy + EVT 能 work（EVO 用 PPO/CPO 做 on-policy + GPD on cumulative cost）。所以这条假设 ruled out。

### 2.5 算力使用情况

| 阶段 | 已花费 | 剩余预算 |
|---|---|---|
| PRISM 三轮迭代实验 | ~150 GPU-h | — |
| 8-cell ablation 10 seed × cartpole_stab | ~60 GPU-h | — |
| Eval（44 min wall-clock × N runs） | ~30 GPU-h | — |
| **当前 budget** | — | **~250 GPU-h（如切到 3080，等价多出 ~150-200 GPU-h）** |

---

## 3. 参考文献与代码 inspiration

### 3.1 核心参考论文（按重要性排序）

#### 3.1.1 EVO (ICML 2025) — **最直接的 baseline**

**Gao et al., "Extreme Value Policy Optimization for Safe Reinforcement Learning"** (ICML 2025)

GitHub: https://github.com/ShiqingGao/EVO

**关键设计**：
- on-policy + GPD（推翻"PPO 与 EVT 不兼容"假设）
- GPD 拟合的对象是 **cumulative cost C**（episode 级别），不是 critic 输出
- Quantile-based constraint：`q_µ + q_H_{ν/(1-µ)} ≤ d`，把 expectation 约束升级成 risk-aware 约束
- off-policy importance resampling 解决 extreme sample 稀缺问题
- Adaptive ν：根据 J_C 自适应调整 exploitation range

**和我们方法的关系**：D-EVO-Choquet 是它的 distributional + multi-source + robust 扩展。

- 但是发现是在omnisafe（https://github.com/PKU-Alignment/omnisafe）上重构的且并不是完整版本代码

#### 3.1.2 EVAC (NeurIPS 2023) — **distributional + EVT 的范式**

**NS et al., "Extreme Risk Mitigation in Reinforcement Learning using Extreme Value Theory"** (NeurIPS 2023)

**关键设计**：
- off-policy actor-critic（DDPG/TD3/SAC 系）
- critic 是 **distributional**（每个 (s,a) 输出 N 个 quantile 值）
- GPD 直接重定义 critic 自身：`Z(s,a) = Z_L(s,a) + (1-η)·Z_H(s,a)`，其中 Z_H 用 GPD 参数化
- η = 0.96，B = 128, K = 100（每 transition 从 GPD 采 K=100 样本做 MLE）
- 不处理 constraint，只做 risk-averse policy

**和我们方法的关系**：distributional critic 设计借鉴 EVAC（quantile critic + GPD on tail quantiles），但 EVAC 不处理 constraint，需要嫁接 EVO 的 constraint 框架。

#### 3.1.3 MICE (ICML 2025) — **诊断 cost underestimation**

**Gao et al., "Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration"** (ICML 2025)

GitHub: https://github.com/ShiqingGao/MICE（注意：和 EVO 是同一个一作）

**关键设计**：
- 诊断了 cost critic 的 underestimation bias（图 2 实测 CPO/PIDLag 在 4 个 env 上系统性低估 cost）
- Flashbulb memory + intrinsic cost：count-based exploration 的 cost-side 镜像
- Adaptive balancing factor β：根据估计 bias 自适应调整 intrinsic cost 强度
- 用 γ^n 衰减保证 intrinsic cost 趋于 0，extrinsic-intrinsic critic 收敛到真值

**和我们方法的关系**：**已决定不直接借鉴**。理由：
- MICE 的 memory mechanism 是 motivation 鲜明的 trick，不是 PRISM/EVO codebase 自然延伸
- random projection + KNN 在低维 state 上效果存疑
- 加进来会让方法变成 6 块组件的拼盘，故事讲不清

但 MICE 的 underestimation 诊断给了我们启发：**distributional critic 可以提供 distribution-level 的 bias detection（KS-test 偏差），比 MICE 的标量 bias ϵ_n 更精确**。这一点可作为论文的 motivation 一句话引用。

#### 3.1.4 FuzRL (NeurIPS 2025) — **Choquet aggregation + B1 bug 起源**

**关键设计**：
- 用 Choquet integral 跨 K 个扰动 shell 聚合
- 原版用 softmax 输出 capacity（导致 Choquet 退化为加权平均，记为 B1 bug）
- 评估扰动协议（FuzRL Table 3）：obs noise + action impulse force + dynamics multiplicative noise 三通道同时作用

**和我们方法的关系**：复用 Choquet 代码（sigmoid + reg 修复版）。FuzRL 的扰动协议作为 robust evaluation 的一部分。

#### 3.1.5 Robust-Gymnasium (ICLR 2025) — **新 main task 环境**

**Gu et al., "Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning"** (ICLR 2025)

GitHub: https://github.com/SafeRL-Lab/Robust-Gymnasium

**关键 features**：
- 60+ task，170+ task variant
- **`RobustSafetyXxx-v4` 系列**：safe RL + 扰动的合体（**完美匹配我们需求**）
  - RobustSafetyAnt-v4, RobustSafetyHalfCheetah-v4, RobustSafetyHopper-v4
  - RobustSafetyWalker2d-v4, RobustSafetySwimmer-v4, RobustSafetyHumanoid-v4
- 扰动通过 `robust_input` dict 注入（modular，不改环境内部代码）
- 支持 obs/act/reward/dynamics 四维扰动 × random/adversarial/set-arbitrarily/semantic 四种模式

**安装**：
```bash
conda create -n robustgymnasium python=3.11
conda activate robustgymnasium
git clone https://github.com/SafeRL-Lab/Robust-Gymnasium
cd Robust-Gymnasium
pip install -r requirements.txt
pip install -e .
```

**接口示例**：
```python
robust_input = {
    "action": action,
    "robust_type": "action",
    "robust_config": args,  # noise_factor, noise_type, noise_sigma 等
}
next_state, reward, terminated, truncated, info = env.step(robust_input)
```

### 3.2 已有 codebase 来源

- **PRISM 主代码**：`evt_otp_fuz_v5_ablation.tar.gz`（基于 v4 + ablation refactor）
- **基础 PPO-Lagrangian**：自己实现，参考 OmniSafe 的设计
- **safe-control-gym**：cartpole_stab / quadrotor_stab 任务来源
- **FuzRL Table 3 评估协议**：自己复现的 obs+act+dyn 三通道扰动

---

## 4. 下一步方向 + 时间成本

### 4.1 当前推荐路径：分阶段 PoC 验证

#### 阶段 0（this week，5 GPU-h，go/no-go 决策点）

**目标**：**确认 robust-gymnasium 接口可用 + GPD 在 RobustSafetyHopper 上拟合可行**。

**两个最小验证脚本**：

**脚本 A（接口适配，3 GPU-h）**：
```bash
# 装 robust-gymnasium
# 把现有 PPO-Lagrangian 跑通 RobustSafetyHopper-v4（无扰动，1M steps）
# 确认 env.step(robust_input) 接口能正常返回 cost
# 确认训练曲线收敛到合理 return
```

**脚本 B（GPD 拟合验证，2 GPU-h）**：
```bash
# 训练完后，收集 1000 episode 的 cumulative cost C
# 拟合 GPD + Gaussian
# 跑 Kolmogorov-Smirnov test，要求 KS(GPD) < 0.1（参照 EVO Figure 7）
# 不达标的话考虑换 task 或调整阈值
```

**Go/no-go 标准**：
- 接口适配 ≤ 1 周 + GPD KS-test 通过 → 全量投入 D-EVO-Choquet
- 接口超过 1 周或 GPD 拟合差 → fallback 到 fuz B1 fix 投 TMLR/workshop

#### 阶段 1（week 2-3，30-50 GPU-h）

**目标**：实现 distributional cost critic，跑 5 seed × 5 setting × 1 task。

**代码工作量**：
- `safe_rl/modules/quantile_critic.py`（QR-style critic，~250 LOC）
- `safe_rl/modules/multi_source_gpd.py`（multi-source GPD heads，~100 LOC，复用现有 GPD 代码）
- `ppo_lagrangian.py` patch（用 Choquet-aggregated CVaR(η) 替代标量 V_C，~50 LOC）

**5 个 setting**：

| setting | 含义 | 来源 |
|---|---|---|
| baseline | PPO-Lagrangian | 已有 |
| EVO-style | 单 GPD + 标量 V_C | 复现 EVO |
| **Dist-EVO** | 单 GPD + quantile V_C | **新** |
| **Dist-EVO-Choquet** | multi-GPD + Choquet + quantile V_C | **新（ours）** |
| Fuz-PPO-Lag (sigmoid) | 复用 fuz B1 fix | 已有 |

**关键决策点**：
- Dist-EVO vs EVO-style：distributional critic 单独是否有用？
- Dist-EVO-Choquet vs Dist-EVO：multi-GPD + Choquet 是否带来增量？

#### 阶段 2（week 4-6，100-150 GPU-h）

**主战场**：RobustSafety MuJoCo 三层环境策略

| 层 | task | seed × method | GPU-h |
|---|---|---|---|
| 1 (debug+appendix) | cartpole_stab | 已有 | 0 |
| 2 (main) | RobustSafetyHopper + HalfCheetah | 5 × 5 × 2 | ~100-150 |
| 3 (generalize) | RobustSafetyAnt | 3 × 5 × 1 | ~30-50 |
| **总计** | | | **~150-200 GPU-h** |

#### 阶段 3（week 7+）

写论文 + 起草专利。Cartpole 数据进 appendix 作为 sanity check。

### 4.2 备选路径（fallback）

**如果阶段 0 不通过**：直接写 fuz B1 fix paper 投 TMLR / 国内 C 刊。

- 已有 cartpole 10 seed 数据 + ablation 完整
- 加 quadrotor 10 seed 验证 generalization（已经在跑或将跑）
- 故事："我们发现 FuzRL 的 B1 bug + OT 在 standard control 上系统性有害"
- 时间：3-4 周写完
- 落点：TMLR / NeurIPS Reproducibility Track / 中文 C 刊

**如果阶段 1 distributional 不显著好**：
- 还是写 fuz B1 fix paper，distributional 作为"我们试过的扩展"放 limitations
- 别死磕，fallback 不丢人

### 4.3 各方向可能性评估

| 方向 | 成功率 | 落点 venue | 时间 |
|---|---|---|---|
| **D-EVO-Choquet 完整实现 + Robust-Gym main result** | 60% | 二三区 SCI / 国内 A 类会议 | 6-8 周 |
| **D-EVO-Choquet on safety-gym (无 robust)** | 75% | 三区 SCI / IEEE 普通会议 | 4-6 周 |
| **fuz B1 fix paper（fallback）** | 90% | TMLR / 国内 C 刊 | 3-4 周 |
| **distributional 失败 + 回到 fuz B1 fix** | — | 同上 | + 1-2 周 |

---

## 5. 几个明确的禁区与注意

### 5.1 不要做的事

1. **不要在论文里把 fuz_softmax 作为 baseline 直接对比 D-EVO-Choquet**——这是隐瞒 B1 fix 的 framing，是 selection bias 问题。正确做法：baseline = Fuz-PPO-Lag (sigmoid implementation)，appendix 一句话提 softmax variant
2. **不要继续在 cartpole_stab 上做 main result**——它太简单，GPD 拟合无意义
3. **不要把 OT 模块带进 D-EVO-Choquet**——已有 4/4 对照证明它在 cartpole 上有害，在 main task 上是已知风险
4. **不要在 PoC 阶段就上 distributional + multi-source + Choquet 三件套**——先做 Dist-EVO（单 GPD + quantile critic）确认 distributional 单独 work，再加 multi-source 和 Choquet

### 5.2 容易踩的坑

1. **Robust-Gymnasium 的 API 是新 gymnasium 1.0**，和老 gym 不兼容。迁移现有 codebase 需要 1 周专门做接口适配
2. **MuJoCo 训练成本远高于 cartpole**：1e7 steps × ~0.5 GPU-h/1M ≈ 5 GPU-h/run。算力预算必须严格控制 seed 数
3. **GPD MLE 在 small batch 上方差大**：参考 EVAC 用 K=100 样本/batch，PPO 的 rollout buffer (2000 steps/epoch) 里 cost > threshold 的样本可能不到 50 个。可能需要 importance resampling（参考 EVO §4.3）
4. **Distributional critic 的 quantile 数选择**：EVAC 用 N=51。太多会增加显存压力，太少 GPD 拟合不准。建议从 N=51 起步

### 5.3 服务器迁移建议

- **从 4×4090 (6r/h × 4) 切换到 3080 (~1.5-2r/h)**
- 单卡显存峰值预估 < 8 GB（PPO + 51 quantile critic + GPD heads 不大）
- 如果 robust-gymnasium 的 MuJoCo 渲染吃显存，先在 3080 上跑 1M step 验证 nvidia-smi 峰值
- 省下的钱 (~1000r+) 等价于多 1-2 个 task 的实验，是直接的论文 quality 提升

---

## 6. 跨窗口讨论建议

### 6.1 单窗口 vs 多窗口

**建议单窗口**。理由：
- 理论决策（IQN vs QR-DQN vs C51）直接影响代码结构（网络架构、loss、训练循环）
- 跨窗口同步 context 成本高
- Claude 的 memory 限于单窗口内

例外：长篇推导（收敛性证明、variance bound）可以单开。

### 6.2 开新窗口起手 prompt 模板

```
我有一个 distributional risk-aware safe RL 项目要继续讨论。
请阅读附件 PROJECT_HANDOFF.md，这是项目完整背景。

我现在想做的具体事是 [填具体动作，例如]：
- "先实现 distributional cost critic 的 PoC"
- "讨论 quantile critic 用 IQN 还是 QR-DQN"  
- "适配 robust-gymnasium 的 step 接口到现有 codebase"
- "审稿人会怎么 attack 我的 D-EVO-Choquet method，帮我提前做 rebuttal 准备"

算力情况：3080 单卡（已迁移），剩余预算 ~250 GPU-h。
时间线：[填具体期限，例如 "下周三见导师" 或 "6 月底前定稿"]。

[可选：附上当前已有数据 / 实验结果 / 错误日志]
```

### 6.3 特别建议

1. **不要让我猜你想做什么**。即便已读 handoff，没有 explicit "我现在要做 X" 我会犹豫
2. **如果已有技术选型偏好**（"用 QR-DQN 不用 IQN"），开头直接说，省得我列对比表
3. **遇到实验结果反直觉时**，把数据贴给我而不是只描述，我能基于数字给具体建议
4. **对话变长后主动让我做 handoff doc**，避免上下文丢失（就像现在）

---

## 7. 附录：关键文件清单

### 7.1 必带到下个窗口

- `PROJECT_HANDOFF.md` (本文档)
- `evt_otp_fuz_v5_ablation.tar.gz`（codebase）
- `CHANGELOG.md`（evaluate_scg + plot_results 的修改记录）
- 三个核心参考 paper PDF：EVO, EVAC, MICE

### 7.2 重要文件路径（codebase 内）

```
evt_otp_fuz_v4/
├── safe_rl/
│   ├── algos/ppo_lagrangian.py      # 主算法（已支持 3 个 orthogonal flag）
│   ├── modules/
│   │   ├── fuzzy_net.py             # Choquet (sigmoid + reg, B1 fixed)
│   │   ├── gpd_head.py              # 现有 GPD MLE 模块（标量版）
│   │   ├── ot_perturb.py            # OT shells（确认要丢弃）
│   │   ├── actor_critic.py          # 标量 V/V_C critic（要替换为 quantile）
│   │   └── buffer.py                # rollout buffer
│   └── envs/
│       ├── safe_cartpole_simple.py  # debug 用，不改
│       └── scg_wrapper.py           # safe-control-gym wrapper（要新增 robust-gymnasium wrapper）
├── scripts/
│   ├── train.py                     # 训练入口
│   ├── evaluate_scg.py              # 三通道 eval（已对齐 FuzRL Table 1）
│   ├── plot_results.py              # 阴影带绘图
│   └── run_ablation_cartpole.sh     # 8-cell ablation launcher
└── V5_CHANGELOG.md                  # 最近一次 refactor 的说明
```

### 7.3 待新建文件（阶段 1）

```
safe_rl/modules/
├── quantile_critic.py               # QR-style cost critic（~250 LOC）
└── multi_source_gpd.py              # multi-source GPD heads（~100 LOC）

safe_rl/envs/
└── robust_gym_wrapper.py            # Robust-Gymnasium adapter（~150 LOC）

scripts/
└── run_dist_evo_hopper.sh           # 阶段 1 的 5 seed × 5 setting launcher
```

---

## 末尾：一句话总结

**当前最优路径**：3080 切换 → 5 GPU-h PoC（接口 + GPD 拟合）→ go: 实施 D-EVO-Choquet @ RobustSafety MuJoCo（6-8 周，~200 GPU-h）→ no-go: fallback 到 fuz B1 fix paper @ TMLR（3-4 周）。

**最差 outcome**：保住 fuz B1 fix paper + 1 个专利 + 完整毕业论文。
**最好 outcome**：D-EVO-Choquet 在 RobustSafety 主流 task 上 SOTA + 二三区 SCI / 国内 A 类会议接收。

下个窗口见。
