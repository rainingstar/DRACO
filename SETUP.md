# DRACO 环境配置详细指南

> 这份文档假设你**完全没有配过环境**。每一步都列出了完整命令、可能遇到的报错以及对应解法。预计跟完一遍：30-60 分钟（取决于网速和是否要装 safe-control-gym）。

---

## 0. 总览：你最终需要什么

DRACO 跑起来需要：

1. **Python 3.10 或 3.11**（不要用 3.12+，gymnasium / safe-control-gym 兼容问题多）
2. **PyTorch 2.x**（CPU 版能跑 toy task；GPU 版才是实战训练用）
3. **基础依赖**：numpy, scipy, gymnasium, matplotlib, pyyaml
4. **可选**：safe-control-gym（用于 cartpole_stab / quadrotor_stab；toy task 不需要）

---

## 1. 选你的硬件路径

|你的环境|推荐方案|
|---|---|
|3080 Linux 工作站|Conda + CUDA 11.8 PyTorch（章节 2.A）|
|4060 Laptop + WSL2 (Ubuntu)|Conda + CUDA 11.8 PyTorch（同 2.A，但 WSL2 一些 GUI 依赖装不上，无关紧要）|
|没 GPU 只想看跑通|Conda + CPU PyTorch（章节 2.B）|
|公司 Mac / 不想装 conda|venv + CPU PyTorch（章节 2.C）|

---

## 2. 步骤一：装 Python + 创建虚拟环境

### 2.A 用 Conda（推荐，最省事）

如果还没装 conda：

```bash
# Linux / WSL2
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
# 一路按 yes，装完后重启 shell（exit + 重新打开）
```

创建 DRACO 专用环境：

```bash
conda create -n draco python=3.11 -y
conda activate draco
# 之后 EVERY 次开新 shell 跑 DRACO 都要先 `conda activate draco`
```

### 2.B Conda CPU-only

同上，但下面装 torch 时选 CPU 版。

### 2.C Venv（不用 conda）

```bash
python3.11 -m venv ~/venvs/draco
source ~/venvs/draco/bin/activate
# 之后 EVERY 次开新 shell 都要 `source ~/venvs/draco/bin/activate`
```

---

## 3. 步骤二：装 PyTorch

**这一步必须按顺序做，先于 requirements.txt**。原因：torch 的 CUDA 版本和系统驱动必须匹配，没法用普通 pip 通配。

### 3.A GPU 路径（3080 / 4060 等）

先看你的驱动支持哪个 CUDA：

```bash
nvidia-smi
# 看右上角 "CUDA Version: XX.X"
# - 如果是 11.x，下面用 cu118
# - 如果是 12.x，下面用 cu121
```

装：

```bash
# CUDA 11.8 路径（推荐，兼容性最好）
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1 路径
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

验证：

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')"
```

预期输出：
```
torch: 2.2.0+cu118
cuda available: True
device: NVIDIA GeForce RTX 3080
```

如果 `cuda available: False`：
- 重启电脑后再试
- 检查 `nvidia-smi` 能输出（说明驱动 ok）
- 可能装错了 cuda 版本，pip uninstall torch 重装

### 3.B CPU 路径

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

---

## 4. 步骤三：装 DRACO 主依赖

```bash
cd /path/to/draco_safe_rl   # 替换成你 unzip 后的目录
pip install -r requirements.txt
```

如果遇到 `ModuleNotFoundError` 或网络超时，挨个装：

```bash
pip install numpy scipy gymnasium matplotlib pyyaml tensorboard tqdm
```

中国大陆用户加镜像加速：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 5. 步骤四：跑 toy task 验证（**3 分钟内完成，最重要**）

不依赖 safe-control-gym 的最小验证：

```bash
cd /path/to/draco_safe_rl
python tests/smoke_test_full.py
```

预期输出（最末几行）：

```
SMOKE TEST SUMMARY
======================================================================
Method       Passed   Elapsed    EpRet      EpCost     Lambda    
baseline     OK          0.5s    -...
fuz          OK          2.2s    -...
evo_style    OK          0.6s    -...
dist_only    OK          1.4s    -...
draco        OK          3.6s    -...
5/5 methods passed.
```

**如果看到 `5/5 methods passed`，恭喜，DRACO 主体完全 work。** 接下来的步骤只是为了用 safe-control-gym 真任务而装。

---

## 6. 步骤五（可选）：装 safe-control-gym（cartpole_stab / quadrotor_stab）

> 跳过这步你也可以拿 toy task 做演示，但中期答辩用真 task 更有说服力。预计 15-30 分钟。

### 6.1 装系统依赖（只在 Linux / WSL 需要）

```bash
sudo apt-get update
sudo apt-get install -y libffi-dev libgl1-mesa-glx libglib2.0-0
```

WSL2 用户：如果 `apt-get` 一直 hang，先 `sudo nano /etc/resolv.conf` 把 `nameserver` 改成 `8.8.8.8`。

### 6.2 装 safe-control-gym

```bash
# 先装它的依赖
pip install pybullet munch dict-deep casadi

# 再装本体（必须 git clone 后 pip install -e，因为 PyPI 上没有最新版）
cd ~                     # 选你喜欢的位置
git clone https://github.com/utiasDSL/safe-control-gym.git
cd safe-control-gym
pip install -e .
```

如果 git clone 太慢：

```bash
# 用 GitHub 镜像
git clone https://gh-proxy.com/https://github.com/utiasDSL/safe-control-gym.git
```

### 6.3 验证 safe-control-gym

```bash
cd /path/to/draco_safe_rl
python -c "
from draco.envs.scg_wrapper import SCGWrapper
env = SCGWrapper(task_name='cartpole_stab', seed=0)
obs, _ = env.reset(seed=0)
print('obs shape:', obs.shape, 'action_space:', env.action_space.shape)
for _ in range(5):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    print(f'  reward={r:.3f}, cost={info.get(\"cost\", 0)}')
env.close()
print('SCG OK')
"
```

预期：5 行打印每行一个 reward + cost，最后输出 `SCG OK`。

如果看到 `Could not load module ... pybullet`，多半是 pybullet 装的版本和 Python 不匹配。试 `pip install pybullet==3.2.5`。

### 6.4 用 SCG task 跑 DRACO 一小段

```bash
python scripts/train.py --task cartpole_stab --method draco \
    --seed 0 --total-steps 20000 --n-envs 4 --log-dir /tmp/draco_test
```

预计 5-10 min。看到 `[ep 1 ... EpRet=...] [ep 2 ...]` 流动并且最后 `DONE` 就是成功了。

---

## 7. 常见坑速查表

| 报错 | 原因 | 解法 |
|---|---|---|
| `CUDA out of memory` | 网络太大 / batch 太大 | 加 `--n-envs 4` 和 `--steps-per-epoch 1024` |
| `Choquet sumG = NaN` | GPD shape 跑飞 | 已经在代码里 clip 了；如果还是出现请提交 issue 附上完整日志 |
| `ModuleNotFoundError: draco` | 没在项目根目录运行 | `cd /path/to/draco_safe_rl` 再跑 |
| SCG 装不上 / pybullet 报错 | 系统包缺失 | 走 toy task 路径，足够答辩用 |
| 训练几个 epoch 后 EpRet 不变 | cost_limit 太低，policy 完全卡死 | 调大 `--cost-limit`（cartpole 试 25→50） |
| Lambda 一直涨到很大 | cost 一直超预算，penalty_lr 太大 | 调小 `--penalty-lr 0.01` |
| `gymnasium.error.Error: Action ...` | act_space 边界对不上 | 检查 SCG 版本，必要时调用 `--task` 时传 normalized_rl_action_space |

---

## 8. 离线打包（如果你要拷到内网机器）

```bash
# 在有网机器上：
pip download -r requirements.txt -d ./offline_pkgs
# 加 torch
pip download torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu -d ./offline_pkgs

# 拷贝 ./offline_pkgs/ 和整个 draco_safe_rl/ 目录到内网机器后：
pip install --no-index --find-links ./offline_pkgs -r requirements.txt
pip install --no-index --find-links ./offline_pkgs torch
```

---

## 9. 我装完了，现在干什么？

跳到 **`README.md` § Quickstart** 跑你的第一个完整训练，或者直接：

```bash
# 跑 5 method × 1 task × 1 seed 的最小验证（10-20 min）
SEEDS=0 TOTAL_STEPS=200000 bash scripts/run_midterm_experiments.sh cartpole_stab

# 看结果
ls runs/cartpole_stab/
python scripts/plot_results.py --runs-dir runs --out test.png
```

如果还有问题，去看：
- `README.md` — 算法介绍 + quickstart
- `EXPERIMENTS.md` — 详细的实验复现协议
- `ALGORITHM.md` — 数学和伪代码
