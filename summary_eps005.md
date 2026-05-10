## Eval @ eps=0.05 on quadrotor_stab

Mean ± std across seeds. n=#seeds with valid eval data.

### Return

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=3) | 262.3±17.6 | 269.7±14.4 | 264.5±15.6 | 260.8±20.3 |
| fuz (n=3) | 161.9±138.3 | 163.5±133.9 | 160.8±137.1 | 163.0±138.6 |
| evo_style (n=3) | 124.5±113.5 | 132.3±123.5 | 135.1±116.3 | 130.9±116.4 |
| dist_only (n=3) | 204.4±92.2 | 202.8±99.6 | 206.4±91.6 | 204.6±94.8 |
| draco (n=3) | 49.9±22.9 | 51.5±25.1 | 53.4±25.8 | 48.2±21.1 |
| **Δ(draco - baseline)** | -212.5 | -218.2 | -211.1 | -212.6 |

### Cost

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=3) | 0.0±0.0 | 0.0±0.0 | 0.3±0.4 | 0.4±0.7 |
| fuz (n=3) | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| evo_style (n=3) | 2.5±3.3 | 1.2±0.3 | 2.1±2.4 | 1.0±1.4 |
| dist_only (n=3) | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.3±0.6 |
| draco (n=3) | 0.1±0.1 | 0.1±0.1 | 0.1±0.1 | 0.0±0.1 |

### GO/No-go signal

- ❌ **obs**: DRACO 51.5±25.1 ≤ baseline 269.7±14.4 (Δ=-218.2) — LOSS
- ❌ **act**: DRACO 53.4±25.8 ≤ baseline 264.5±15.6 (Δ=-211.1) — LOSS
- ❌ **dyn**: DRACO 48.2±21.1 ≤ baseline 260.8±20.3 (Δ=-212.6) — LOSS

**Verdict: PIVOT to P1 (obs-side augmentation)** — only 0 stat-sig wins of 3 channels
