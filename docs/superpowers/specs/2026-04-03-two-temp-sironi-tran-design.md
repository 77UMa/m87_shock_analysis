# Two-Temperature Sironi-Tran Electron Heating Design

## Goal

Replace the current beta-closure downstream electron-temperature branch with a physically closed single chain based on:

1. upstream `R-beta` electron-ion partition,
2. adiabatic downstream electron compression,
3. Sironi & Tran (2024) super-adiabatic shock heating,
4. reinjection of the resulting `Theta_e2` into the full nonthermal injection chain (`p_min`, `K_inj`, `f_e`, `N_inj`, `gamma_min`).

The target branch is `physics/two-temp-sironi-tran`.

## Scientific scope

This is a first-principles-facing physics replacement, not an A/B software feature experiment.

- The new chain becomes the only active downstream electron-heating chain.
- We do **not** keep the old beta-closure as a runtime fallback or comparison mode.
- We do **not** preserve the previous split where `C_grid/UNTH` and `gamma_min` use different temperature semantics.
- We do **not** change the ipole C-side interface semantics in this phase.

## Motivation

The maintained engineering guide documents that the current pipeline still carries a split design:

- `C_grid/UNTH` is produced by a legacy empirical injection interface,
- `gamma_min` is produced by a newer physical branch.

That split is no longer acceptable for this branch because the user explicitly wants physical logical closure. The new design therefore makes `Theta_e2` the unique thermodynamic driver for both:

- thermal quantities entering the injection formulae,
- and the nonthermal lower cutoff `gamma_min`.

The literature review identifies Sironi & Tran (2024) as the most relevant compact closure for trans-relativistic standing shocks in black-hole accretion/jet environments, and specifically recommends the form

\[
T_{e2} = T_{e2,ad}\left[1 + 0.0016 M_s^{3.6}\right]
\]

with the important requirement that `M_s` is the relativistic sonic Mach number rather than the fast magnetosonic Mach number.

## Current code constraints

### Existing shock-side state

`src/core/shock_v1.py::find_shocks_in_roi_mhd()` already computes, for accepted SRMHD mainline shock cells:

- upstream sampled state: `rho1`, `press1`, `bsq1`, `w1`, `u_n1`,
- downstream sampled state: `rho2`, `press2`, `bsq2`,
- fast-mode diagnostics: `sr_mach_normal`, `cfast_n_upstream`, `theta_Bn`,
- sampling validity diagnostics: `sample_boundary_clipped_grid`, `verified_mask`, `sampling_stats`.

However, the current exported result dict does **not** expose enough upstream thermodynamic state for a clean upstream-electron-temperature calculation, and it does **not** expose relativistic sonic Mach.

### Existing NT-side logic

`src/core/nt_electron_v1.py::calculate_nonthermal_electrons()` currently:

- computes `q` from `mainline_mach`,
- computes `p_min`, `K_inj`, `f_e`, `N_inj` from legacy `downstream_temp/downstream_n_e`,
- computes `gamma_min` from a separate beta-closure branch using `beta2`, `press2/rho2`, and `r_low/r_high/beta_crit`.

This is the split that will be removed.

## Target architecture

### Module boundary

#### `src/core/shock_v1.py`
Responsibility remains limited to shock detection and upstream/downstream fluid-state export.

It will:
- continue to own candidate screening, gradient marching, and acceptance,
- continue to compute accepted mainline shock masks,
- newly export the upstream and downstream fluid fields required by the new electron-heating model,
- newly compute and export relativistic sonic Mach.

It will **not** implement the Sironi-Tran heating prescription itself.

#### `src/core/nt_electron_v1.py`
This becomes the unique location for electron thermodynamics and nonthermal injection.

It will:
- compute upstream electron temperature from the upstream fluid state,
- compute adiabatic downstream electron compression,
- apply the Sironi-Tran super-adiabatic correction,
- derive `gamma_min`, `p_min`, `K_inj`, `f_e`, `e_thermal`, and `N_inj` from the same `Theta_e2`.

#### `src/workflows/workflowFull_v2.py`
Workflow only passes configuration and logs diagnostics. No direct array physics is added here.

#### `src/workflows/base_workflow.py`
HDF5 writing remains an interface/export layer. It may export additional diagnostics, but it does not interpret the new physics.

## New shock-to-NT interface

For every accepted shock cell, `find_shocks_in_roi_mhd()` must export these additional arrays:

- `rho1_code_grid`
- `press1_code_grid`
- `bsq1_code_grid`
- `beta1_grid`
- `rho2_code_grid` (already present)
- `press2_code_grid` (already present)
- `sr_sonic_mach` or equivalently named `sr_sonic_mach_grid`

The recommended naming is `sr_sonic_mach` in the result dict if consistency with existing scalar names is desired, or `sr_sonic_mach_grid` if the file follows explicit grid naming. The final implementation should choose one name and use it consistently everywhere.

### Sonic Mach definition

The new Mach used by the heating prescription must be the relativistic sonic Mach, not the current fast-mode Mach.

For accepted upstream sampled states:

- compute relativistic sound speed from upstream enthalpy:

\[
c_s^2 = \frac{\gamma P_1}{w_1}
\]

- then compute sonic four-velocity:

\[
u_s = \frac{c_s}{\sqrt{1-c_s^2}}
\]

- then compute relativistic sonic Mach from the upstream normal four-velocity:

\[
M_s = \frac{|u_{n1}|}{u_s}
\]

This quantity is the only Mach that feeds the Sironi-Tran heating law.

## Electron-heating physics chain

Within `calculate_nonthermal_electrons()`, for active shock cells:

### Step 1: upstream electron partition

Use upstream beta:

\[
R_1 = 80 \frac{\beta_1^2}{1+\beta_1^2} + \frac{1}{1+\beta_1^2}
\]

and upstream thermodynamic ratio:

\[
\Theta_{e1} = \left(\frac{P_1}{\rho_1 c^2}\right)\left(\frac{m_p}{m_e}\right)\frac{1}{1+R_1}
\]

Implementation note: to avoid reviving the old unit-chain inconsistency, the code should compute `press1_over_rho1` from the same stable code-unit thermodynamic ratio style used elsewhere, then convert to dimensionless electron temperature in one controlled place.

### Step 2: adiabatic downstream compression

For relativistic electrons with adiabatic index `4/3`:

\[
\Theta_{e2,ad} = \Theta_{e1}\left(\frac{\rho_2}{\rho_1}\right)^{1/3}
\]

### Step 3: Sironi-Tran super-adiabatic correction

\[
\Theta_{e2} = \Theta_{e2,ad}\left[1 + 0.0016 M_s^{3.6}\right]
\]

where `M_s` is the relativistic sonic Mach defined above.

### Step 4: gamma_min

\[
\gamma_{min} = 1 + 3\Theta_{e2}
\]

### Step 5: reinject into the full injection chain

The resulting `Theta_e2` is the sole thermodynamic temperature variable for:

- `T_e2 = Theta_e2 * m_e c^2 / k_B`
- `p_min = sqrt(2 x_inj^2 Theta_e2)`
- `K_inj`
- `f_e(p_min)`
- `e_thermal`
- `xi_lin`
- `N_inj / C_grid`

This removes the current split where `p_min/K_inj/f_e/N_inj` and `gamma_min` come from different thermal assumptions.

## Configuration changes

The active NT configuration should explicitly contain the Sironi-Tran fit parameters:

- `sironi_tran_coeff = 0.0016`
- `sironi_tran_exp = 3.6`

These live in `config["nt_params"]`.

The previous beta-closure control parameters are no longer active mainline physics for this branch:

- `r_low`
- `r_high`
- `beta_crit`

Preferred behavior: remove them from active defaults and fail loudly if someone tries to use them as if the old closure were still live. Silent coexistence is not acceptable on this branch.

## Failure policy

The new chain keeps explicit failure accounting rather than silently degrading.

Recommended failure codes:

- `ok`
- `press1_nonpositive`
- `rho1_nonpositive`
- `press2_nonpositive`
- `rho2_nonpositive`
- `beta1_invalid`
- `sonic_mach_invalid`
- `boundary_clipped`
- `theta_e1_invalid`
- `theta_e2_invalid`
- `gamma_min_le_one`

Priority rule:
- `boundary_clipped` should take precedence over downstream thermodynamic interpretation failures, because it means the sampled interface state itself is geometrically unreliable.

Fallback behavior:
- invalid `gamma_min` is still written as `1.0`, but the failure code must make the reason explicit.
- the implementation should avoid inventing a second hidden thermal fallback chain.

## Logging and diagnostics

### Shock-side diagnostics

Add active-shock logging for:

- `shock.rho1_code.active`
- `shock.press1_code.active`
- `shock.beta1.active`
- `shock.sr_sonic_mach.active`

### NT-side diagnostics

Add active-shock logging for:

- `nt.beta1_shocks`
- `nt.R1_shocks`
- `nt.theta_e1_shocks`
- `nt.theta_e2_ad_shocks`
- `nt.sironi_boost_shocks`
- `nt.theta_e2_shocks`
- `nt.Te2_shocks`
- `nt.p_min_shocks`
- `nt.gamma_min_failure`

Codepath strings should move from the old beta-closure wording to explicit two-temperature/Sironi wording, for example:

- `Electron heating branch | upstream R-beta + adiabatic compression + Sironi-Tran boost`
- remove wording such as `Gamma-min beta closure` from the active path.

## HDF5 export

The ipole-facing required datasets remain unchanged in this phase:

- `UNTH`
- `p`
- `GAMMA_MIN`

To support debugging and scientific inspection, the design recommends adding these diagnostic exports when available:

- `RHO1_CODE`
- `PRESS1_CODE`
- `BETA1`
- `SONIC_MACH`
- `THETA_E1`
- `THETA_E2_AD`
- `THETA_E`
- `P_MIN_PHYSICAL`

This makes it possible to separate compression-driven heating from Sironi boost in later analysis.

## Validation strategy

This branch is validated as a physics replacement, not by preserving old numbers.

### Code-level validation

1. Confirm accepted shock cells have finite positive sonic Mach.
2. Confirm sonic Mach is distinct from fast magnetosonic Mach.
3. Confirm the chain
   - `beta1 -> R1 -> Theta_e1 -> Theta_e2_ad -> Theta_e2 -> gamma_min`
   is finite and interpretable over active shock cells.
4. Confirm the injection chain now consumes the same `Theta_e2` without split semantics.

### Physics diagnostics

Inspect active shock-cell distributions for:

- `beta1`
- `R1`
- `M_s`
- `Theta_e1`
- `Theta_e2_ad`
- `Theta_e2 / Theta_e2_ad`
- `gamma_min`
- `N_inj`

Relative to the current beta-closure branch, the key expected shift is that `Theta_e2` and `gamma_min` should no longer collapse near the old cold-electron floor unless the shock truly predicts that.

### End-to-end validation

Remote verification should use:

- single-snapshot HDF5 generation,
- HDF5 diagnostic inspection,
- `compare_models` checks on the resulting Model C behavior.

Comparison against the previous branch is diagnostic only: the purpose is to explain the impact of the new physics, not to preserve the old chain.

## Out-of-scope items

This design does not include:

- parallel support for old beta-closure and new Sironi heating,
- advection-chain refactoring,
- ipole C-side semantic changes,
- broader unit-system redesign outside the needed controlled thermodynamic ratios.

## Recommended implementation shape

Use a single-chain replacement with minimal module-boundary disruption:

- `shock_v1.py`: export fluid states + relativistic sonic Mach,
- `nt_electron_v1.py`: own the full electron-heating and injection chain,
- `workflowFull_v2.py`: pass config and log,
- `base_workflow.py`: export diagnostics.

This keeps the physical architecture clean while staying aligned with the existing engineering boundaries in the maintained guide.
