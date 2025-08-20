# Experiment/Test Plan: VAT Threshold & Rate Reforms

This document specifies a code-agnostic experiment suite to measure firm- and system-level impacts of UK VAT threshold and rate reforms. It is designed to be executed by a coding agent (language/framework agnostic).

---

## 1) Core Objects (agent inputs)

**Firm archetype (per firm `i`):**
- `sic`: e.g., 96020 hairdressing, 47190 retail, 41201 construction, etc.
- `turnover_baseline`: pre-policy turnover (excl. VAT).
- `b2b_share` ∈ [0,1]: share of sales to VAT-registered firms (recover input VAT).
- `input_vat_share` ∈ [0,1]: share of inputs bearing reclaimable VAT.
- `gross_margin` ∈ [0,1]: pre-VAT gross margin on sales.
- `labor_intensity` ∈ {low, med, high}: used for split-rate targeting.
- `region` (optional).

**Behavioural parameters:**
- `pass_through` ∈ [0,1]: share of VAT change passed to consumer prices.
- `demand_elasticity`: price elasticity of demand (typically −0.2 to −1.2; services toward 0).
- `bunching_elasticity`: responsiveness of *reported* turnover near the threshold (e.g., 0.05–0.30).
- `bunching_band`: turnover window around the threshold within which bunching can occur (e.g., £20k).

**Policy scenario:**
- `standard_rate` (e.g., 0.20)
- `reduced_rate` (e.g., 0.10 for labour-intensive sectors)
- `threshold_design` ∈ {`"notch"`, `"taper"`}
- If `"notch"`: `T` (registration threshold).
- If `"taper"`: `T_start`, `T_full` (liability increases linearly from `T_start` to full rate at `T_full`).
- `split_rate_target`: rule for applying `reduced_rate` (e.g., `labor_intensity='high'` and `sic` in list).

**Simulation flags:**
- `mode` ∈ {`static`, `behavioural`}.
- `horizon_years`: number of years to project (optional).
- `turnover_growth_baseline`: baseline growth rate if projecting.

---

## 2) Mechanics (agent should implement)

1. **VAT liability calculation** per firm under scenario:
   - Determine applicable output VAT rate(s) by sector (split-rate rules).
   - Output VAT on **B2C** portion only (B2B customers reclaim).
   - Input VAT credit if registered, approximated by `input_vat_share * inputs`.
   - **Notch**: zero VAT if unregistered; full VAT if registered.
   - **Taper**: VAT liability scales linearly between `T_start` and `T_full`.

2. **Behavioural response**:
   - **Price change**: `ΔP/P = pass_through * ΔVAT_rate_on_B2C`.
   - **Quantity change (B2C)**: `%ΔQ = demand_elasticity * %ΔP`.
   - **Bunching near threshold (reported turnover)**:
     - If `"notch"` and `turnover_baseline ∈ [T−bunching_band, T+bunching_band]`, reduce **reported** turnover toward `T−ε` using `bunching_elasticity`.
     - If `"taper"`, apply only a fraction of that adjustment (e.g., multiply by `(1 − taper_intensity)` where `taper_intensity` reflects the slope between `T_start` and `T_full`).

3. **Outputs per firm**:
   - `turnover_post` (reported, post-behaviour).
   - `VAT_liability` and `effective_vat_rate = VAT_liability / turnover_post`.
   - `operating_profit_change` (after VAT & behaviour).
   - `registered` (boolean).
   - Static vs behavioural decomposition of all deltas.

4. **Aggregations**:
   - By **SIC**, **size band** (<£65k, £65–£90k, £90–£110k, >£135k), **labour_intensity**, **B2B share terciles**.
   - Totals: VAT receipts, # registered, distribution of effective VAT rates.

---

## 3) Scenario Set (to run against all archetypes)

- **S0 Baseline:** 20% standard rate; **notch** at **£90k**.
- **S1 Higher threshold:** notch at **£100k**.
- **S2 Taper A:** linear taper **£65k → £110k** to full 20%.
- **S3 Taper B:** linear taper **£90k → £135k** to full 20%.
- **S4 Split-rate:** 10% for labour-intensive SICs; 20% others; baseline threshold.
- **S5 Split-rate + Taper B.**

---

## 4) Firm Archetypes (diagnostic grid)

- **A1 Hairdresser (B2C, high labor):** turnover = {£80k, £89k, £91k, £105k}; `b2b_share=0.05`, `pass_through=0.7`, `demand_elasticity=-0.4`, `bunching_elasticity=0.20`, `input_vat_share=0.2`.
- **A2 Bicycle repair (B2C, high labor):** same grid; `pass_through=0.6`, `demand_elasticity=-0.5`, `bunching_elasticity=0.15`, `input_vat_share=0.2`.
- **A3 Cafe (B2C, medium labor, higher inputs):** turnover = {£85k, £95k, £120k}; `input_vat_share=0.4`, `pass_through=0.8`, `demand_elasticity=-0.8`, `bunching_elasticity=0.10`.
- **A4 Small contractor (mixed B2B/B2C):** turnover = {£88k, £100k, £130k}; `b2b_share=0.6`, `pass_through=0.5`, `demand_elasticity=-0.3`, `bunching_elasticity=0.10`, `input_vat_share=0.3`.
- **A5 B2B services (accounting):** turnover = {£85k, £95k, £140k}; `b2b_share=0.95`, `pass_through=0.3`, `demand_elasticity=-0.2`, `bunching_elasticity=0.05`, `input_vat_share=0.2`.

Run **static** and **behavioural** for all (S0–S5 × A1–A5).

---

## 5) Expected Qualitative Results (assertions to check)

- **S1 vs S0 (raise threshold):** fewer firms register just above old £90k; VAT receipts ↓ in B2C sectors; bunching mass shifts toward **£100k**.
- **S2/S3 vs S0 (tapers):** registration cliff becomes smooth; **bunching metric** (excess mass in `[T−band, T]`) falls; receipts near band may ↑ (more partial liability); profit discontinuity shrinks.
- **S4 (split-rate):** price pressure and VAT receipts ↓ in labour‑intensive B2C sectors; demand reductions smaller.
- **S5:** combines S3 and S4 effects: minimal bunching + lower consumer price pressure in targeted SICs.
- **B2B-heavy firms:** minimal quantity change; effective burden low under all scenarios.

---

## 6) Metrics & Diagnostics

- **Firm-level:** ΔVAT (static & behavioural), Δprofit, registration flip, `effective_vat_rate`.
- **Bunching metric:** `Excess = (count in [T−band, T]) − expected count from local linear fit` (within ±band).
- **Continuity metric:** jump in after‑tax profit from `T−ε` to `T+ε` (should shrink under tapers).
- **Sensitivity sweeps:** `pass_through ∈ {0.4, 0.7, 1.0}`, `demand_elasticity ∈ {−0.3, −0.6, −0.9}`, `bunching_elasticity ∈ {0.05, 0.15, 0.30}`; report tornado table for VAT receipts and bunching metric.

---

## 7) Edge / Robustness Tests

- **All‑B2B firm** (`b2b_share=1`): receipts ~0; results invariant to rate changes (except admin/inputs).
- **All‑B2C + zero input VAT:** largest consumer‑side effects; monotonicity: higher rate ⇒ non‑negative receipts and non‑increasing quantities/profits.
- **Tiny firm far below T:** invariant to threshold design in static mode.
- **Huge firm far above T_full:** taper vs notch identical for liability.
- **No pass‑through (0)** and **full pass‑through (1)** extremes.
- **Elasticity = 0:** behavioural outputs equal static outputs.

---

## 8) Recommended Outputs

- Table: firm × scenario with {registered, VAT, Δprofit, effective rate}.
- Plot: after‑tax profit vs turnover around threshold for S0/S2/S3 (cliff vs smooth).
- Histogram: turnover distribution showing bunching under S0 and reduction under S2/S3.
- Bar chart: VAT receipts by SIC across S0–S5.
- Tornado table: sensitivity sweep for A1 (hairdresser).

---

## 9) Acceptance Criteria (per run)

- **Continuity:** Profit jump at threshold S2/S3 < S0.
- **Bunching:** Excess mass below threshold S2/S3 < S0.
- **Incidence:** B2B‑heavy A5 burden ≪ B2C A1/A3.
- **Monotonicity:** Increasing pass‑through or |elasticity| does not increase quantities or profits, holding rates constant.

---

*End of spec.*
