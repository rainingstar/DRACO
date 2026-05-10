## Eval @ eps=0.20 on quadrotor_stab

Mean ± std across seeds. n=#seeds with valid eval data.

### Return

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=5) | 265.0±13.9 | 263.9±11.4 | 260.0±14.4 | 248.7±25.1 |
| fuz (n=5) | 206.6±115.5 | 208.1±114.1 | 201.9±113.4 | 201.7±113.2 |
| evo_style (n=5) | 184.4±114.8 | 203.6±109.9 | 186.2±110.1 | 172.5±108.1 |
| dist_only (n=5) | 183.6±101.7 | 179.9±103.4 | 175.7±100.4 | 173.3±97.6 |
| draco (n=5) | 142.5±127.8 | 135.3±131.0 | 139.6±128.4 | 136.6±123.1 |
| **Δ(draco - baseline)** | -122.5 | -128.6 | -120.5 | -112.1 |

### Cost

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=5) | 1.3±2.5 | 4.6±8.9 | 2.5±3.9 | 6.2±9.2 |
| fuz (n=5) | 0.0±0.0 | 0.3±0.5 | 0.0±0.0 | 0.2±0.5 |
| evo_style (n=5) | 1.8±2.6 | 1.0±1.1 | 1.2±1.0 | 2.3±4.6 |
| dist_only (n=5) | 3.7±6.8 | 1.7±2.4 | 2.0±2.9 | 5.7±5.9 |
| draco (n=5) | 0.1±0.1 | 0.6±1.0 | 0.2±0.5 | 1.3±2.9 |

### GO/No-go signal

- ❌ **obs**: DRACO 135.3±131.0 ≤ baseline 263.9±11.4 (Δ=-128.6) — LOSS
- ❌ **act**: DRACO 139.6±128.4 ≤ baseline 260.0±14.4 (Δ=-120.5) — LOSS
- ❌ **dyn**: DRACO 136.6±123.1 ≤ baseline 248.7±25.1 (Δ=-112.1) — LOSS

**Verdict: PIVOT to P1 (obs-side augmentation)** — only 0 stat-sig wins of 3 channels
