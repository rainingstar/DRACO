# DRACO Algorithm Reference

> Math, pseudocode, and design rationale. Cross-references `draco/algos/draco.py` line numbers in brackets.

---

## 1. Problem setting

Constrained MDP `M = (S, A, r, c, P, γ, μ)`:

- standard reward `r: S × A → R`
- cost `c: S × A → R≥0` (safety violation indicator or scalar)
- policy `π_θ` parameterized by θ
- discount γ for both reward and cost

Standard PPO-Lagrangian objective:

$$\max_\theta J_R(\theta) - \lambda \cdot J_C(\theta), \quad J_R = \mathbb{E}_\pi[\sum_t \gamma^t r_t],\ J_C = \mathbb{E}_\pi[\sum_t \gamma^t c_t]$$

with constraint `J_C ≤ d` (cost budget).

**DRACO replaces `J_C` with a risk-aware, multi-source, Choquet-aggregated cost summary**:

$$J_C^{\text{DRACO}}(\pi) = \mathbb{E}_{s_0 \sim \mu}[V_C^{\text{robust}}(s_0)]$$

where `V_C^robust` is computed via the pipeline below.

---

## 2. Five components

### 2.1 IQN distributional cost critic [`iqn_critic.py`]

For each state `s`, instead of a scalar `V_C(s)`, we maintain `Z_C(s, τ): S × [0,1] → R`, an approximation of `F_C^{-1}(τ | s)` (cost-to-go quantile function).

**Architecture** [`IQNCostCritic.forward`, lines 92-119]:

```
phi(s)     = MLP(s)                                  (state encoder, dim F)
psi(τ)     = ReLU(W · cos(π · i · τ) + b)            (cosine basis, i=1..n_cos=64)
Z_C(s, τ)  = softplus( head( phi(s) ⊙ psi(τ) ) )      (Hadamard product → MLP head)
```

**Training loss** (quantile Huber regression, [`quantile_huber_loss`, lines 130-159]):

$$\mathcal{L}_{\text{IQN}} = \mathbb{E}_{(s, c, s')} \mathbb{E}_{\tau, \tau'} \left[ \rho_\kappa^\tau\left( z^* - Z_C(s, \tau) \right) \right]$$

where the target `z* = G_t = sum_{k≥t} γ^{k-t} c_k` (Monte-Carlo cost-to-go from the rollout).

In practice we sample N=32 prediction τ and broadcast the scalar MC target across N=16 target τ' (see `_update_iqn_cost_critic`). This is a degenerate IQN (target distribution is a Dirac at `G_t`), but works empirically because (a) MC `G_t` is a single sample of `F_C^{-1}(U | s)` for some random U, so quantile Huber over many τ still pushes Z toward correct conditional quantiles; (b) full distributional Bellman target (sample s', forward Z_C through, etc.) is more accurate but ~2× slower per gradient step.

### 2.2 Multi-source perturbation shells [`draco.py:compute_vc_robust`, lines 220-285]

For each state `s`, we generate K=4 perturbed states:

$$s_k = s + \varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0, \sigma_k^2 I)$$

with `σ_k ∈ {0.01, 0.05, 0.10, 0.20}` (stratified magnitudes). Each shell `k` is a "risk source" — physically interpretable as "what if observation noise / dynamics drift were of magnitude σ_k?".

This replaces the OT-perturbation shells of PRISM (which were learned and shown to be harmful) with random Gaussian shells (which preserve the multi-source structure for free).

### 2.3 Per-state per-shell GPD tail [`multi_source_gpd.py`]

For each `(s_k, shell_index k)`:

1. Sample `N=64` quantile fractions `τ` uniformly, get `Z_C(s_k, τ)` from IQN.
2. Compute the data-driven threshold `u_k = min{Z_C(s_k, τ_j) : τ_j > η}` (the η-quantile of cost-to-go), with η=0.7 [`fit_threshold_and_excesses`].
3. Pass `(s_k, k)` through `MultiSourceGPDHead` to get `(ξ_k, β_k)`.
4. Closed-form CVaR: `CVaR_k = u_k + β_k / (1 - ξ_k)`  [`gpd_cvar`].

**GPD MLE training** [`_update_gpd`]: for each rollout step, the K-shell tail samples `{Z_C(s_k, τ_j) - u_k : τ_j > η}` provide ~10 excesses per (state, shell). Maximize log-likelihood:

$$\log f_{\xi, \beta}(y) = -\log\beta - (1/\xi + 1)\log(1 + \xi y / \beta), \quad y \geq 0$$

**Numerical stability note** [`gpd_log_likelihood`, lines 127-145]: at ξ=0 the formula has `1/ξ = ∞`. We use a straight-through estimator: forward pass uses `ξ_safe = sign(ξ) · max(|ξ|, 10⁻³)`, backward pass uses raw `ξ`. This lets gradients flow through ξ=0 without producing NaN.

### 2.4 Choquet aggregation [`fuzzy_net.py`]

Given K CVaR estimates `{CVaR_1, ..., CVaR_K}` and a state-conditional capacity `g(s) = (g_1, ..., g_K)`:

$$V_C^{\text{robust}}(s) = -\text{Choquet}(g(s), \{-\text{CVaR}_k\}) = \sum_{i=1}^K \text{CVaR}_{(i)} \cdot \left[ m(A_i) - m(A_{i+1}) \right]$$

where `(i)` is the permutation that sorts `CVaR` ascending (so the dual emphasizes the maximum), `A_i = {(i),...,(K)}`, and `m` is the Sugeno fuzzy measure satisfying:

$$\prod_{k=1}^K (1 + \lambda \cdot g_k) = 1 + \lambda$$

(solved by Newton-Raphson, [`_solve_lambda`]).

**Capacity parameterization** [`ChoquetAggregator`]: `g_k(s) = sigmoid(MLP_k(s))`, with non-additivity regularizer:

$$R_{\text{na}}(g) = (\sum_k g_k - 1.2)^2$$

(target `1.2 = K · 0.3 = 4 · 0.3`, pushing capacity sub-additive → emphasizes worst sources).

### 2.5 Lagrangian update [`draco.py:update`, lines 425-440]

We update the dual via SGD on the raw parameter `p`:

$$p \leftarrow p + \alpha \cdot (J_C^{\text{est}}(\pi) - d), \quad \lambda = \text{softplus}(p)$$

where `J_C^est` is the running mean of episode costs (logger-tracked). SGD instead of Adam because Adam normalizes the gradient, decoupling step size from violation magnitude (this was bug B3 in the PRISM v2→v3 fixes).

---

## 3. Full pseudocode

```python
# Per epoch:
for epoch in 1..N_epochs:
    # ============ ROLLOUT (parallel N envs × T steps) ============
    for t in 1..T:
        for each env in parallel:
            a_t ~ pi(s_t)
            v_t   = V_R(s_t)                              # scalar reward critic
            vc_t  = compute_vc_robust(s_t)                # ⭐ risk-aware summary
            store (s_t, a_t, r_t, c_t, v_t, vc_t)
            s_{t+1} = step(env, a_t)

    # GAE on reward and cost
    A_R, R_R   = GAE(r, v, γ_R, λ_R)
    A_C, R_C   = GAE(c, vc_robust, γ_C, λ_C)
    G_t        = MC cost-to-go (no GAE; single sample of F_C^{-1}|s)

    # ============ POLICY UPDATE ============
    for k in 1..K_pi:
        L_pi = -E[ ratio · A_R - λ · ratio · A_C ] - β · entropy
        if KL > 1.5 · KL_target: break
        SGD step on θ_pi

    # ============ CRITIC UPDATES ============
    L_VR = MSE(V_R(s) - R_R)                              # reward critic
    L_ZC = quantile_huber(Z_C(s, τ), G_t over τ)         # IQN cost critic [draco mode]
    L_GPD = -E_k[log p(excesses_k | ξ_k, β_k)]            # multi-source GPD
    L_Cho = α · (sum_k g_k - 1.2)²                        # capacity reg

    # ============ LAGRANGIAN UPDATE ============
    p ← p + α_p · (avg_episode_cost - d)
    λ ← softplus(p)

# Return: trained π, V_R, Z_C, GPD heads, Choquet capacity
```

---

## 4. Method-specific behavior of `compute_vc_robust`

For unifying the 5 baselines in one codebase:

|method|`compute_vc_robust(s)`|
|---|---|
|`baseline`|`V_C(s)` (scalar critic)|
|`fuz`|`Choquet({V_C(s + ε_k)})` (multi-shell scalar critic)|
|`evo_style`|`V_C(s) + β_0 / (1 - ξ_0)` (single GPD on scalar V_C)|
|`dist_only`|`empirical_CVaR_η(Z_C(s))` (IQN, single source, no GPD)|
|**`draco`**|`Choquet({u_k + β_k / (1 - ξ_k)})` (full pipeline)|

This mapping is implemented in `compute_vc_robust` via the (use_iqn, use_evt, n_shells, use_choquet) flag matrix.

---

## 5. Complexity / compute footprint

Per gradient step (cartpole_stab, n_envs=8, batch=2048):

|component|FLOPs (approx)|wall time on 3080|
|---|---|---|
|Actor + V_R update (PPO)|~5 GFLOPs|0.5s|
|IQN critic update (32 quantiles × 80 iters)|~8 GFLOPs|0.8s|
|Multi-shell rollout for `compute_vc_robust`|~3 GFLOPs / rollout step|negligible|
|GPD MLE update|~0.5 GFLOPs|0.1s|
|Choquet update|~0.05 GFLOPs|0.05s|
|**Total per epoch (2048 steps + update)**| |~3-4s|

For 1M steps total: ~500 epochs × 3s = **25 min** on 3080. Quadrotor_stab roughly 1.5× this.

---

## 6. Hyperparameter sensitivity

|parameter|default|sensitivity|notes|
|---|---|---|---|
|`n_shells K`|4|⚠ moderate|K=2: degenerate Choquet; K=8: 254 capacity params, hard to train|
|`shell_sigmas`|0.01-0.20|⚠ moderate|Should span "small" to "large" relative to obs scale|
|`eta` (GPD threshold)|0.7|🟢 low|0.5-0.9 all work; 0.7 gives ~10 tail samples|
|`n_tau_train`|32|🟢 low|16 also OK; 64 marginally better|
|`cost_limit d`|25.0|🔴 high|Task-specific; tune from baseline behavior|
|`penalty_lr α_p`|0.05|🔴 high|Too large: λ oscillates; too small: slow convergence|
|`nonadditive_target`|0.3 (per dim)|🟢 low|0.2-0.4 all give sub-additive Choquet|

---

## 7. Known limitations (be honest in the paper)

1. **MC cost-to-go is a degenerate quantile target** — full distributional Bellman target (sample s', forward Z_C(s', τ'), etc.) would be more accurate. We chose MC for simplicity; consider switching to full distributional update if results are weak.

2. **Choquet capacity can collapse to additive** — if the regularizer coefficient is too low, optimization may push `sum(g) → 1` (additive). Monitor `ChoquetSumG` in logs and increase `choquet_reg_coef` if it drifts toward 1.0.

3. **GPD MLE on small tail** — at the start of training, many states have ξ ≈ 0 and the GPD CVaR ≈ u + β. The "tail behavior" really only kicks in after the IQN has learned a non-trivial quantile spread (~50K env steps).

4. **Shell σ values are hand-tuned** — they could in principle be learned (as in OT shells), but the PRISM ablation showed learning shells *hurts*. We keep them fixed.

5. **No theoretical convergence guarantee** — like all PPO-Lag methods, convergence is empirical. Choquet aggregation does not add new theoretical risk; the aggregation is bounded between min and max of the input CVaRs, so the cost summary is always finite.
