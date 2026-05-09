# DRACO Experiment Replication Guide

> 中期答辩用。给定 220 GPU-h 预算，目标 5/15 完成。

---

## 0. Schedule (倒排)

|日期|任务|GPU-h|
|---|---|---|
|**Day 1 (5/9)** ← 今天|环境配置 + smoke test 跑通 + 启动 cartpole_stab 5 method × 3 seed| 3-5 |
|**Day 2 (5/10)**|完成 cartpole 训练 (~15 GPU-h)，启动 quadrotor 训练，做 cartpole 评估| 25-30 |
|**Day 3 (5/11)**|完成 quadrotor 训练 (~30-40 GPU-h)，做评估| 35-40 |
|**Day 4 (5/12)**|生成所有图表 / 表格，开始写答辩 slide| 5-10 |
|**Day 5 (5/13)**|查漏补缺：补 seed？补 task？调 hyperparameter？| 10-30 |
|**Day 6 (5/14)**|finalize slides + 演练答辩| 0 |
|**Day 7 (5/15)**|答辩| — |

预计实际消耗：**80-120 GPU-h**，留 100+ GPU-h 给突发情况（再跑一组 seed / 改 hp / 跑 robust-gymnasium）。

---

## 1. Pre-flight checklist

```bash
# 在跑大实验前，强制走完：
cd /path/to/draco_safe_rl

# (a) 确认环境干净
python -c "import torch, numpy, gymnasium, scipy; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# (b) Smoke test 必须 5/5 pass
python tests/smoke_test_full.py
# 如果不是 5/5 pass，停下来排查，不要继续

# (c) SCG 真任务能跑（如果选择了 cartpole_stab/quadrotor_stab）
python -c "from draco.envs.scg_wrapper import SCGWrapper; \
e = SCGWrapper(task_name='cartpole_stab', seed=0); \
e.reset(); print('SCG OK', e.observation_space.shape)"

# (d) 一个 mini-run 跑通完整流程（20K steps，~3 min）
python scripts/train.py --task cartpole_stab --method draco \
    --seed 0 --total-steps 20000 --n-envs 4 --log-dir /tmp/draco_dryrun --save-model

# (e) 评估能跑
python scripts/evaluate.py --task cartpole_stab \
    --model-path /tmp/draco_dryrun/cartpole_stab/draco/*.model.pt \
    --eps 0.10 --n-eval-episodes 5
```

如果以上 5 个全部通过，进入正式实验。

---

## 2. 实验矩阵

|task|methods|seeds|total_steps|approx wall-time / run|
|---|---|---|---|---|
|cartpole_stab|baseline, fuz, evo_style, dist_only, draco|0, 1, 2|1M|~25 min on 3080|
|quadrotor_stab|baseline, fuz, evo_style, dist_only, draco|0, 1, 2|1.5M|~50 min on 3080|

总：**5 × 3 × 2 = 30 runs**, ~30 GPU-h 训练 + ~5 GPU-h 评估。

### 启动命令

```bash
# 自动按 task × method × seed 顺序跑 + 评估 + 出图：
bash scripts/run_midterm_experiments.sh

# 或者只跑一个 task：
bash scripts/run_midterm_experiments.sh cartpole_stab

# 或者控制 seed 数量：
SEEDS="0 1" bash scripts/run_midterm_experiments.sh cartpole_stab
```

---

## 3. 监控训练

每个 run 进行时，另开一个 shell：

```bash
# 实时看最新一条 log
tail -f runs/cartpole_stab/draco/cartpole_stab_draco_seed0.txt

# 或一行命令看所有 run 的最后状态
for f in runs/*/*/*.txt; do echo "=== $f ==="; tail -1 "$f"; done
```

**关键监控指标**（每个 epoch 都打印）：

|指标|健康范围|不健康表现|可能原因|
|---|---|---|---|
|`EpRet`|cartpole 应稳步上升从 ~20 到 ~100+|永远 < 30 或上下震荡|policy 太保守 / cost_limit 太严|
|`EpCost`|从 ~50 降到 cost_limit 附近|永远 > 100|env 任务太难 / lambda 涨不动|
|`Lambda`|稳步上升到 1-10 之间|< 0.1（permissive） 或 > 100（暴涨）|`penalty_lr` 调|
|`LossPi`|0.05-1.0，时降时升正常|NaN / inf|学习率太大，加 grad clip|
|`LossVc`|稳步下降|不变 / 上升| critic 没在学|
|`LossGPD`|从 +large 降到接近 0|永远 NaN|GPD MLE 跑飞，看 `GPD_xi_mean`|
|`GPD_xi_mean`|cartpole 上 0-0.3，quadrotor 0.1-0.5|>0.9 或 <-0.4|tail 太短/太长，调 η|
|`ChoquetSumG`|1.0-1.4 (target 1.2)|→1.0|reg 系数太低|

如果遇到异常，**先查 `LossPi` 是否 NaN**——这是其它 issue 的连锁起点。

---

## 4. 评估协议 (FuzRL Table 3 风格)

每个训出来的 model 做 4 通道评估：

|channel|含义|eps=0.10 时的具体值|
|---|---|---|
|`clean`|无扰动|—|
|`obs`|加 obs 维度的 Gaussian 噪声|std=0.10 on `[0,1,2,3]` (cartpole) / `[0,1,4,5,7,10]` (quadrotor)|
|`act`|加 action impulse force|F0 ~ U[-1, 1], 持续 80 steps，offset 20，decay 0.9|
|`dyn`|inertial param 乘性 jitter|relative std 0.10 on POLE_LENGTH / POLE_MASS (cartpole) 或 MASS / IYY (quadrotor)|

**自动化评估命令**（已经被 `run_midterm_experiments.sh` 调用）：

```bash
python scripts/evaluate.py \
    --model-path runs/cartpole_stab/draco/cartpole_stab_draco_seed0.model.pt \
    --task cartpole_stab \
    --eps 0.10 --n-eval-episodes 30 --seed 1000
```

输出：`<run>.eval.json`，包含 4 个 channel 各自的 `(ret_mean, ret_std, cost_mean, cost_std)`。

---

## 5. 分析与报表

### 5.1 训练曲线

```bash
python scripts/plot_results.py \
    --runs-dir runs --tasks cartpole_stab quadrotor_stab \
    --metrics EpRet/mean EpCost/mean Lambda/mean \
    --out midterm_curves.png
```

输出 3×2 grid（3 metrics × 2 tasks），每个子图 5 条曲线（5 method）+ 阴影带（seed 间方差）。

### 5.2 评估表格（FuzRL Table 1 风格）

```bash
python scripts/aggregate_eval.py --runs-dir runs --out midterm_eval_table.csv
```

输出长表（task × method × seed × channel）+ summary 表（task × method × channel，aggregate over seeds）。

### 5.3 答辩 slide 必带表格模板

| Method | Cartpole-Stab Ret(clean) | Cost(clean) | Ret(obs) | Cost(obs) | Ret(act) | Cost(act) | Ret(dyn) | Cost(dyn) |
|---|---|---|---|---|---|---|---|---|
| Baseline | xx.x±x.x | x.xx±x.xx | ... | ... | ... | ... | ... | ... |
| Fuz | ... | ... | ... | ... | ... | ... | ... | ... |
| EVO-style | ... | ... | ... | ... | ... | ... | ... | ... |
| Dist-Only | ... | ... | ... | ... | ... | ... | ... | ... |
| **DRACO (ours)** | ... | ... | ... | ... | ... | ... | ... | ... |

**期望结果**（基于 PRISM 实验 + theoretical reasoning）：
- DRACO `Cost(*)` 应该**全列都 ≤ Baseline**
- DRACO `Ret(clean)` 与 Fuz 接近（不一定要严格 dominate clean Ret）
- DRACO `Ret(obs/act/dyn)` 应该**比 Baseline / Fuz / EVO-style 更稳健**（std 更低）
- Dist-Only 在 clean 应该和 baseline 类似，在 perturbed 上略好

如果上面有任何一条**严重违反**，需要排查（hp 没调好 / 实现 bug / task 不适合）。

---

## 6. Fallback 路径

### 6.1 如果某个 method 训练发散

```bash
# 单独重训一个 run，强制小学习率：
python scripts/train.py --task cartpole_stab --method draco --seed 0 \
    --pi-lr 1e-4 --v-lr 5e-4 --vc-lr 5e-4 --penalty-lr 1e-2 \
    --total-steps 1000000 --save-model --log-dir runs_safer
```

### 6.2 如果 GPU 时间紧张

降到 200K steps + 2 seeds + 只用 cartpole_stab：

```bash
SEEDS="0 1" TOTAL_STEPS=200000 bash scripts/run_midterm_experiments.sh cartpole_stab
```

预计 ~5 GPU-h 总计。够画一个 5-method 比较图。

### 6.3 如果 quadrotor 真不收敛

paper handoff 已经预计到这个风险。fallback 是**只用 cartpole_stab + 多个 cost_limit 设置**模拟难度梯度：

```bash
for D in 10 25 50 100; do
    python scripts/train.py --task cartpole_stab --method draco \
        --cost-limit $D --seed 0 --total-steps 1000000 --save-model \
        --run-name cartpole_stab_draco_d${D}_seed0
done
```

这构成"safety budget sweep"，单 task 也能讲故事。

### 6.4 保底方案：toy task only

```bash
# toy task 跑得飞快（CPU 都行），5 method × 3 seed × 200K steps = ~30 min
SEEDS="0 1 2" TOTAL_STEPS=200000 bash scripts/run_midterm_experiments.sh toy
```

toy task 不能做 perturbation eval，但可以展示 **5 method 的 learning curve 区分度**——这对 "证明算法 work" 已经足够了。

---

## 7. 答辩前 checklist

- [ ] `tests/smoke_test_full.py` 5/5 pass 在最终 codebase 上
- [ ] `runs/` 目录下每个 (task, method, seed) 都有 `*.jsonl` 和 `*.model.pt`
- [ ] 每个 model 都做了 4-channel eval（toy 除外）
- [ ] `midterm_curves.png` 看起来 DRACO 至少在某些 metric 上 dominant
- [ ] `midterm_eval_table.csv` 数字都不是 NaN
- [ ] PPT slide 包含：
  - [ ] 算法 motivation（risk-aware 三层 + 多源）
  - [ ] DRACO 流程图（用 ALGORITHM.md 的伪代码截图）
  - [ ] 5-method 对比表（cartpole + quadrotor）
  - [ ] 训练曲线（4 method 阴影带）
  - [ ] 与 EVO 的 differentiation table（看上次回复的对比表）
  - [ ] limitation 一页（诚实，避免被攻击）

---

## 8. 复盘 / 下一步（写在最后）

实验完成后，立即做：

```bash
# 备份所有 logs 和 models
tar -czf draco_runs_$(date +%Y%m%d).tar.gz runs/ midterm_*.png midterm_*.csv

# 把当前 codebase 也打包，避免改到一半丢东西
tar -czf draco_codebase_$(date +%Y%m%d).tar.gz draco_safe_rl/
```

然后开始写论文 / 专利。
