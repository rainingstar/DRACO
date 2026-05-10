## Eval @ eps=0.10 on quadrotor_stab

Mean ± std across seeds. n=#seeds with valid eval data.

### Return

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=5) | 265.0±13.9 | 270.7±8.5 | 264.2±17.4 | 256.8±17.3 |
| fuz (n=5) | 206.6±115.5 | 208.3±115.0 | 204.7±115.2 | 206.3±114.3 |
| evo_style (n=5) | 184.4±114.8 | 198.7±111.3 | 189.7±113.0 | 186.4±111.2 |
| dist_only (n=5) | 183.6±101.7 | 187.4±105.4 | 179.8±102.7 | 184.5±105.5 |
| draco (n=5) | 142.5±127.8 | 139.5±130.0 | 142.5±127.5 | 142.7±125.8 |
| **Δ(draco - baseline)** | -122.5 | -131.2 | -121.6 | -114.2 |

### Cost

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=5) | 1.3±2.5 | 3.1±6.1 | 1.0±1.8 | 2.1±4.2 |
| fuz (n=5) | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| evo_style (n=5) | 1.8±2.6 | 0.8±1.1 | 1.5±1.3 | 2.8±4.7 |
| dist_only (n=5) | 3.7±6.8 | 2.6±4.4 | 2.9±4.6 | 3.7±4.9 |
| draco (n=5) | 0.1±0.1 | 0.2±0.2 | 0.1±0.2 | 0.5±0.7 |

### GO/No-go signal

- ❌ **obs**: DRACO 139.5±130.0 ≤ baseline 270.7±8.5 (Δ=-131.2) — LOSS
- ❌ **act**: DRACO 142.5±127.5 ≤ baseline 264.2±17.4 (Δ=-121.6) — LOSS
- ❌ **dyn**: DRACO 142.7±125.8 ≤ baseline 256.8±17.3 (Δ=-114.2) — LOSS

**Verdict: PIVOT to P1 (obs-side augmentation)** — only 0 stat-sig wins of 3 channels
