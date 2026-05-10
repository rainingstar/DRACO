## Eval @ eps=0.10 on quadrotor_stab

Mean ± std across seeds. n=#seeds with valid eval data.

### Return

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=5) | 265.0±13.9 | 270.7±8.5 | 264.2±17.4 | 256.8±17.3 |
| fuz (n=5) | 206.6±115.5 | 208.3±115.0 | 204.7±115.2 | 206.3±114.3 |
| evo_style (n=5) | 184.4±114.8 | 198.7±111.3 | 189.7±113.0 | 186.4±111.2 |
| dist_only (n=5) | 183.6±101.7 | 187.4±105.4 | 179.8±102.7 | 184.5±105.5 |
| draco (n=10) | 179.1±119.2 | 179.0±121.2 | 179.5±118.9 | 177.2±116.5 |
| **Δ(draco - baseline)** | -85.8 | -91.7 | -84.7 | -79.7 |

### Cost

| method (n) | clean | obs | act | dyn |
|---|---|---|---|---|
| baseline (n=5) | 1.3±2.5 | 3.1±6.1 | 1.0±1.8 | 2.1±4.2 |
| fuz (n=5) | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| evo_style (n=5) | 1.8±2.6 | 0.8±1.1 | 1.5±1.3 | 2.8±4.7 |
| dist_only (n=5) | 3.7±6.8 | 2.6±4.4 | 2.9±4.6 | 3.7±4.9 |
| draco (n=10) | 1.0±2.7 | 1.5±2.6 | 1.0±2.6 | 1.5±2.5 |

### GO/No-go signal

- ❌ **obs**: DRACO 179.0±121.2 ≤ baseline 270.7±8.5 (Δ=-91.7) — LOSS
- ❌ **act**: DRACO 179.5±118.9 ≤ baseline 264.2±17.4 (Δ=-84.7) — LOSS
- ❌ **dyn**: DRACO 177.2±116.5 ≤ baseline 256.8±17.3 (Δ=-79.7) — LOSS

**Verdict: PIVOT to P1 (obs-side augmentation)** — only 0 stat-sig wins of 3 channels
