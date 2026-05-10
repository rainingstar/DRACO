## Eval @ eps=0.20 on quadrotor_stab

Mean ± std across seeds. n=#seeds with valid eval data.

### Return

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=3) | 262.3±17.6 | 265.2±14.9 | 259.8±20.1 | 247.6±33.0 |
| fuz (n=3) | 161.9±138.3 | 166.1±139.3 | 155.1±132.2 | 156.1±133.4 |
| evo_style (n=3) | 124.5±113.5 | 157.1±126.7 | 130.4±111.8 | 112.6±99.6 |
| dist_only (n=3) | 204.4±92.2 | 198.8±84.8 | 189.5±86.1 | 192.1±85.0 |
| draco (n=3) | 49.9±22.9 | 40.2±20.1 | 46.4±22.2 | 47.3±21.0 |
| **Δ(draco - baseline)** | -212.5 | -225.0 | -213.4 | -200.2 |

### Cost

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=3) | 0.0±0.0 | 0.0±0.0 | 0.8±0.7 | 1.6±2.3 |
| fuz (n=3) | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.4±0.7 |
| evo_style (n=3) | 2.5±3.3 | 0.9±1.0 | 1.7±1.0 | 3.7±5.9 |
| dist_only (n=3) | 0.0±0.0 | 0.3±0.5 | 0.0±0.0 | 4.6±8.0 |
| draco (n=3) | 0.1±0.1 | 0.3±0.5 | 0.1±0.1 | 0.1±0.1 |

### GO/No-go signal

- ❌ **obs**: DRACO 40.2±20.1 ≤ baseline 265.2±14.9 (Δ=-225.0) — LOSS
- ❌ **act**: DRACO 46.4±22.2 ≤ baseline 259.8±20.1 (Δ=-213.4) — LOSS
- ❌ **dyn**: DRACO 47.3±21.0 ≤ baseline 247.6±33.0 (Δ=-200.2) — LOSS

**Verdict: PIVOT to P1 (obs-side augmentation)** — only 0 stat-sig wins of 3 channels
