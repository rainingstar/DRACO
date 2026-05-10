## Eval @ eps=0.10 on quadrotor_stab

Mean ± std across seeds. n=#seeds with valid eval data.

### Return

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=3) | 262.3±17.6 | 273.1±9.6 | 261.0±22.8 | 253.7±23.6 |
| fuz (n=3) | 161.9±138.3 | 164.2±138.4 | 156.8±134.0 | 161.0±135.8 |
| evo_style (n=3) | 124.5±113.5 | 150.1±126.1 | 134.3±118.5 | 128.7±110.6 |
| dist_only (n=3) | 204.4±92.2 | 208.0±89.5 | 201.9±92.3 | 201.1±94.5 |
| draco (n=3) | 49.9±22.9 | 45.4±24.5 | 50.2±23.3 | 51.6±23.4 |
| **Δ(draco - baseline)** | -212.5 | -227.7 | -210.8 | -202.0 |

### Cost

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=3) | 0.0±0.0 | 0.3±0.5 | 0.0±0.1 | 0.2±0.4 |
| fuz (n=3) | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| evo_style (n=3) | 2.5±3.3 | 1.3±1.2 | 1.5±1.1 | 4.0±6.1 |
| dist_only (n=3) | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.5±0.8 |
| draco (n=3) | 0.1±0.1 | 0.3±0.3 | 0.2±0.2 | 0.3±0.3 |

### GO/No-go signal

- ❌ **obs**: DRACO 45.4±24.5 ≤ baseline 273.1±9.6 (Δ=-227.7) — LOSS
- ❌ **act**: DRACO 50.2±23.3 ≤ baseline 261.0±22.8 (Δ=-210.8) — LOSS
- ❌ **dyn**: DRACO 51.6±23.4 ≤ baseline 253.7±23.6 (Δ=-202.0) — LOSS

**Verdict: PIVOT to P1 (obs-side augmentation)** — only 0 stat-sig wins of 3 channels
