## Eval @ eps=0.05 on quadrotor_stab

Mean ± std across seeds. n=#seeds with valid eval data.

### Return

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=5) | 265.0±13.9 | 269.6±11.6 | 266.0±11.4 | 263.7±15.2 |
| fuz (n=5) | 206.6±115.5 | 207.3±112.1 | 207.6±116.2 | 208.7±116.3 |
| evo_style (n=5) | 184.4±114.8 | 189.9±117.7 | 189.7±111.1 | 188.1±113.6 |
| dist_only (n=5) | 183.6±101.7 | 184.3±109.2 | 185.1±102.1 | 186.0±105.1 |
| draco (n=5) | 142.5±127.8 | 143.0±126.5 | 144.1±125.5 | 140.9±127.8 |
| **Δ(draco - baseline)** | -122.5 | -126.7 | -122.0 | -122.8 |

### Cost

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=5) | 1.3±2.5 | 2.7±6.1 | 1.8±2.5 | 1.1±1.9 |
| fuz (n=5) | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| evo_style (n=5) | 1.8±2.6 | 0.7±0.7 | 2.4±2.8 | 0.6±1.1 |
| dist_only (n=5) | 3.7±6.8 | 2.6±4.4 | 3.3±5.7 | 4.0±5.8 |
| draco (n=5) | 0.1±0.1 | 0.2±0.2 | 0.0±0.1 | 0.4±0.8 |

### GO/No-go signal

- ❌ **obs**: DRACO 143.0±126.5 ≤ baseline 269.6±11.6 (Δ=-126.7) — LOSS
- ❌ **act**: DRACO 144.1±125.5 ≤ baseline 266.0±11.4 (Δ=-122.0) — LOSS
- ❌ **dyn**: DRACO 140.9±127.8 ≤ baseline 263.7±15.2 (Δ=-122.8) — LOSS

**Verdict: PIVOT to P1 (obs-side augmentation)** — only 0 stat-sig wins of 3 channels
