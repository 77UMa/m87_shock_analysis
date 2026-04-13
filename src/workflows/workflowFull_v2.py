"""Main DSA workflow for processing a single Athena++ snapshot."""

import os
import time

import numpy as np

from src.utils.logging_config import is_dual_logger, log_error, log_finish, log_start
from src.workflows.base_workflow import calculate_dsa_physics, load_and_slice_data, save_h5_file


def process_snapshot(filename, config, logger=None):
    """Process one snapshot and generate one ipole HDF5 file."""
    input_athdf_file = filename
    base_name = os.path.basename(input_athdf_file).replace(".athdf", "")
    dual_logger = logger if is_dual_logger(logger) else None

    start_time = time.time()
    stage_times = {}

    try:
        if dual_logger:
            dual_logger.human.log_start(base_name, config)
            dual_logger.ai.func_enter("process_snapshot", {"filename": base_name})
            dual_logger.save_config(config)
            dual_logger.ai.debug(
                f"Resolved output roots: metadata={config['output_directory']}, data={config['data_output_directory']}"
            )
        elif logger:
            log_start(logger, base_name, config)
        else:
            print(f">>> Processing {base_name}")

        output_dir = os.path.join(config["data_output_directory"], "ipole_inputs")
        os.makedirs(output_dir, exist_ok=True)
        output_h5 = os.path.join(output_dir, f"{base_name}_dsa_input.h5")
        if dual_logger:
            dual_logger.human.info(f"Large HDF5 output directory: {output_dir}")
            dual_logger.ai.codepath("Output routing", f"ipole_inputs written to data_output_directory: {output_dir}")

        step_start = time.time()
        if dual_logger:
            dual_logger.ai.codepath("Step 1", "load_and_slice_data")
        roi_data = load_and_slice_data(input_athdf_file, config["roi_params"], dual_logger)
        stage_times["load"] = time.time() - step_start

        if roi_data is None:
            msg = f"Failed to load ROI data for {base_name}"
            if dual_logger:
                dual_logger.human.error(msg)
                dual_logger.ai.codepath("Load failure", "roi_data is None")
            elif logger:
                logger.error(msg)
            return False

        if dual_logger:
            dual_logger.human.time("Load ROI data", stage_times["load"])
            dual_logger.ai.data("roi_data.shape", {"rho": roi_data["rho"].shape})

        step_start = time.time()
        if dual_logger:
            dual_logger.ai.codepath("Step 2", "calculate_dsa_physics")

        physics_cfg = config.get("physics", {})
        g_cgs = 6.674e-8
        m_sun = 1.989e33
        c_light_cgs = 2.998e10
        m_unit_val = physics_cfg.get("M_unit", 1e25)
        mbh_solar = physics_cfg.get("MBH_solar", 6.2e9)
        l_unit_val = g_cgs * (mbh_solar * m_sun) / c_light_cgs**2
        rho_unit = m_unit_val / l_unit_val**3

        nt_params = dict(config["nt_params"])
        nt_params.pop("electron_temp_fraction", None)
        nt_params["rho_unit"] = rho_unit
        nt_params["u_unit"] = rho_unit * c_light_cgs**2
        removed_nt_controls = [key for key in ("r_low", "r_high", "beta_crit") if key in nt_params or key in physics_cfg]
        if removed_nt_controls:
            raise ValueError(
                "Removed beta-closure controls detected: "
                + ", ".join(sorted(set(removed_nt_controls)))
                + ". Two-temperature Sironi-Tran heating is now the only active electron-heating chain."
            )
        nt_params.setdefault("sironi_tran_coeff", 0.0016)
        nt_params.setdefault("sironi_tran_exp", 3.6)
        nt_params.setdefault("sironi_tran_delta_max", 3.0)
        nt_params.setdefault("eta_inj_e0", 1.0e-3)
        nt_params.setdefault("eps_nth_e0", 3.0e-3)
        nt_params.setdefault("theta_bn_quench", 50.0)
        nt_params.setdefault("theta_bn_width", 10.0)
        nt_params.setdefault("sonic_mach_inj_min", 1.5)
        nt_params.setdefault("inj_model", "pic_dual_cap")
        nt_params.setdefault("energy_budget_model", "total_internal_energy_excess")
        nt_params.setdefault("p_eff_model", "hybrid_classical_relativistic")
        nt_params.setdefault("classical_fast_mach_max", 1.8)
        nt_params.setdefault("relativistic_fast_mach_min", 3.0)
        nt_params.setdefault("theta_bn_parallel_max", 35.0)
        nt_params.setdefault("theta_bn_oblique_max", 60.0)
        nt_params.setdefault("sigma_rel_parallel_max", 1.0e-3)
        nt_params.setdefault("sigma_rel_oblique_max", 1.0e-2)
        nt_params.setdefault("p_eff_parallel", 2.35)
        nt_params.setdefault("p_eff_oblique", 2.8)
        nt_params.setdefault("p_eff_steep", 3.5)
        nt_params.setdefault("p_eff_floor", 1.5)
        nt_params.setdefault("p_eff_ceiling", 4.5)
        shock_sr_cfg = dict(config.get("shock_sr", {}))
        stale_keys = [key for key in ("enable_sr_refine", "use_sr_refined_mask_for_nt") if key in shock_sr_cfg]
        if stale_keys:
            raise ValueError(
                "Removed shock_sr controls detected: "
                + ", ".join(stale_keys)
                + ". SRMHD is now the only mainline shock chain."
            )
        shock_params = dict(config["shock_params"])
        removed_shock_controls = [
            key for key in ("mach_threshold_loose", "min_physical_mach") if key in shock_params
        ]
        if removed_shock_controls:
            raise ValueError(
                "Removed classical shock controls detected: "
                + ", ".join(removed_shock_controls)
                + ". Candidate screening is now SRMHD-mainline only."
            )
        shock_params.update(
            {
                "sr_mach_min": shock_sr_cfg.get("sr_mach_min", 1.2),
                "jump_residual_max": shock_sr_cfg.get("jump_residual_max", 0.4),
            }
        )
        enable_advection = physics_cfg.get("enable_advection", False)

        if dual_logger:
            dual_logger.ai.debug(
                f"Derived units: L_unit={l_unit_val:.3e}, rho_unit={rho_unit:.3e}, u_unit={nt_params['u_unit']:.3e}, "
                f"sironi_tran_coeff={nt_params['sironi_tran_coeff']:.4f}, sironi_tran_exp={nt_params['sironi_tran_exp']:.2f}, "
                f"sironi_tran_delta_max={nt_params['sironi_tran_delta_max']:.2f}, inj_model={nt_params['inj_model']}, "
                f"energy_budget_model={nt_params['energy_budget_model']}, "
                f"p_eff_model={nt_params['p_eff_model']}, "
                f"eta_inj_e0={nt_params['eta_inj_e0']:.2e}, eps_nth_e0={nt_params['eps_nth_e0']:.2e}"
            )
            dual_logger.ai.codepath("Physics branch", f"enable_advection={enable_advection}")
            dual_logger.human.info(
                "SRMHD mainline config: "
                f"sr_mach_min={shock_params['sr_mach_min']:.2f}, "
                f"jump_residual_max={shock_params['jump_residual_max']:.2f}"
            )
            dual_logger.ai.codepath(
                "SRMHD mainline config",
                f"sr_mach_min={shock_params['sr_mach_min']:.2f}, "
                f"jump_residual_max={shock_params['jump_residual_max']:.2f}",
            )

        shock_props, nonthermal_props = calculate_dsa_physics(
            roi_data,
            shock_params,
            nt_params,
            dual_logger,
        )
        stage_times["dsa"] = time.time() - step_start

        if dual_logger:
            dual_logger.human.time("Shock detection + nonthermal electrons", stage_times["dsa"])

        if dual_logger and np.any(shock_props["mask"]):
            mask = shock_props["mask"]
            mach_vals = shock_props["mainline_mach"][mask]
            sigma_vals = shock_props.get("sigma_grid")
            sigma_vals = sigma_vals[mask] if sigma_vals is not None else None
            coverage = float(np.sum(mask) / mask.size)

            sampling_stats = shock_props.get("sampling_stats", {})
            dual_logger.human.log_stage_shock_acceptance(
                n_shock_cells=int(np.sum(mask)),
                coverage=coverage,
                mach_values=mach_vals,
                sigma_values=sigma_vals,
                sampling_stats=sampling_stats,
            )
            dual_logger.ai.data(
                "stage_summary.shock_acceptance",
                {
                    "shock_count": int(np.sum(mask)),
                    "coverage": coverage,
                    "mainline_mach_median": float(np.median(mach_vals)) if mach_vals.size else 0.0,
                    "mainline_mach_min": float(np.min(mach_vals)) if mach_vals.size else 0.0,
                    "mainline_mach_max": float(np.max(mach_vals)) if mach_vals.size else 0.0,
                    "sigma_median": float(np.median(sigma_vals)) if sigma_vals is not None and sigma_vals.size else 0.0,
                    "verified_count": int(sampling_stats.get("verified_count", 0)),
                    "sr_refined_count": int(sampling_stats.get("sr_refined_count", 0)),
                    "rejected_low_mach": int(sampling_stats.get("sr_rejected_low_mach_count", 0)),
                    "rejected_jump": int(sampling_stats.get("sr_rejected_jump_count", 0)),
                    "rejected_entropy": int(sampling_stats.get("sr_rejected_entropy_count", 0)),
                    "accepted_boundary_clipped": int(sampling_stats.get("boundary_clipped_verified_count", 0)),
                },
            )
        elif dual_logger:
            dual_logger.human.result("Shock statistics: no shock cells detected")
            dual_logger.ai.codepath("No shocks after Step 2", "shock_props.mask contains no true cells")

        step_start = time.time()
        if enable_advection:
            if dual_logger:
                dual_logger.human.info("Running advection-diffusion stage")
                dual_logger.ai.codepath("Step 3", "advection enabled")
            try:
                from src.core.advection_v0 import solve_steady_advection

                nonthermal_props = solve_steady_advection(roi_data, nonthermal_props, config)
                if dual_logger:
                    dual_logger.human.info("Advection-diffusion stage completed")
                    dual_logger.ai.codepath("Advection stage", "completed")
            except ImportError as exc:
                warning_msg = f"Advection module import failed, stage skipped: {exc}"
                if dual_logger:
                    dual_logger.human.warning(warning_msg)
                    dual_logger.ai.codepath("Advection stage", f"import failure: {exc}")
                elif logger:
                    logger.warning(warning_msg)
            except Exception as exc:
                warning_msg = f"Advection stage failed, using original nonthermal properties: {exc}"
                if dual_logger:
                    dual_logger.human.warning(warning_msg)
                    dual_logger.ai.codepath("Advection stage", f"runtime failure: {exc}")
                elif logger:
                    logger.warning(warning_msg)
        elif dual_logger:
            dual_logger.ai.codepath("Step 3", "advection skipped")

        stage_times["advection"] = time.time() - step_start
        if dual_logger and enable_advection:
            dual_logger.human.time("Advection-diffusion", stage_times["advection"])

        if dual_logger and np.any(shock_props["mask"]):
            verified_mask = shock_props.get("verified_mask", shock_props["mask"])
            nt_mask = nonthermal_props.get("mask", shock_props["mask"])
            c_grid = nonthermal_props.get("unth_code_grid", nonthermal_props.get("C_grid", np.array([])))
            q_grid = nonthermal_props.get("q_grid", np.array([]))
            gamma_min_grid = nonthermal_props.get("gamma_min_grid", np.array([]))
            gamma_failure = nonthermal_props.get("gamma_min_failure_code_grid", np.array([]))
            sigma_suppression = nonthermal_props.get("sigma_suppression_grid")
            if np.size(c_grid) > 0:
                c_vals = c_grid[nt_mask]
                q_vals = q_grid[nt_mask]
                p_eff_vals = nonthermal_props.get("p_eff_grid", np.array([]))
                p_eff_vals = p_eff_vals[nt_mask] if np.size(p_eff_vals) > 0 else np.array([])
                gamma_vals = gamma_min_grid[nt_mask] if np.size(gamma_min_grid) > 0 else np.array([1.0])
                gamma_failure_vals = gamma_failure[nt_mask] if np.size(gamma_failure) > 0 else np.array([], dtype=int)
                suppression_vals = sigma_suppression[nt_mask] if sigma_suppression is not None else None
                theta_e1_vals = nonthermal_props.get("theta_e1_grid", np.array([]))
                theta_e1_vals = theta_e1_vals[nt_mask] if np.size(theta_e1_vals) > 0 else np.array([])
                theta_e2_ad_vals = nonthermal_props.get("theta_e2_ad_grid", np.array([]))
                theta_e2_ad_vals = theta_e2_ad_vals[nt_mask] if np.size(theta_e2_ad_vals) > 0 else np.array([])
                theta_e_vals = nonthermal_props.get("theta_e_grid", np.array([]))
                theta_e_vals = theta_e_vals[nt_mask] if np.size(theta_e_vals) > 0 else np.array([])
                sironi_boost_vals = nonthermal_props.get("sironi_boost_grid", np.array([]))
                sironi_boost_vals = sironi_boost_vals[nt_mask] if np.size(sironi_boost_vals) > 0 else np.array([])
                te2_vals = nonthermal_props.get("Te2_grid", np.array([]))
                te2_vals = te2_vals[nt_mask] if np.size(te2_vals) > 0 else np.array([])
                r1_vals = nonthermal_props.get("R1_grid", np.array([]))
                r1_vals = r1_vals[nt_mask] if np.size(r1_vals) > 0 else np.array([])
                beta1_vals = shock_props.get("beta1_grid", np.array([]))
                beta1_vals = beta1_vals[nt_mask] if np.size(beta1_vals) > 0 else np.array([])
                sonic_vals = shock_props.get("sr_sonic_mach", np.array([]))
                sonic_vals = sonic_vals[nt_mask] if np.size(sonic_vals) > 0 else np.array([])
                theta_bn_vals = shock_props.get("theta_Bn", np.array([]))
                theta_bn_vals = theta_bn_vals[nt_mask] if np.size(theta_bn_vals) > 0 else np.array([])
                inj_gate_vals = nonthermal_props.get("inj_gate_grid", np.array([]))
                inj_gate_vals = inj_gate_vals[nt_mask] if np.size(inj_gate_vals) > 0 else np.array([])
                eta_vals = nonthermal_props.get("eta_inj_e_grid", np.array([]))
                eta_vals = eta_vals[nt_mask] if np.size(eta_vals) > 0 else np.array([])
                eps_vals = nonthermal_props.get("eps_nth_e_grid", np.array([]))
                eps_vals = eps_vals[nt_mask] if np.size(eps_vals) > 0 else np.array([])
                p_min_vals = nonthermal_props.get("p_min_physical_grid", np.array([]))
                p_min_vals = p_min_vals[nt_mask] if np.size(p_min_vals) > 0 else np.array([])
                n_nth_vals = nonthermal_props.get("n_nth_phys_grid", np.array([]))
                n_nth_vals = n_nth_vals[nt_mask] if np.size(n_nth_vals) > 0 else np.array([])
                n_nth_eta_vals = nonthermal_props.get("n_nth_eta_phys_grid", np.array([]))
                n_nth_eta_vals = n_nth_eta_vals[nt_mask] if np.size(n_nth_eta_vals) > 0 else np.array([])
                n_nth_eps_vals = nonthermal_props.get("n_nth_eps_phys_grid", np.array([]))
                n_nth_eps_vals = n_nth_eps_vals[nt_mask] if np.size(n_nth_eps_vals) > 0 else np.array([])
                e_diss_e_vals = nonthermal_props.get("e_diss_e_grid", np.array([]))
                e_diss_e_vals = e_diss_e_vals[nt_mask] if np.size(e_diss_e_vals) > 0 else np.array([])
                e_diss_tot_vals = nonthermal_props.get("e_diss_tot_grid", np.array([]))
                e_diss_tot_vals = e_diss_tot_vals[nt_mask] if np.size(e_diss_tot_vals) > 0 else np.array([])
                u_nth_budget_vals = nonthermal_props.get("u_nth_budget_grid", np.array([]))
                u_nth_budget_vals = u_nth_budget_vals[nt_mask] if np.size(u_nth_budget_vals) > 0 else np.array([])
                energy_budget_model = nonthermal_props.get("energy_budget_model", nt_params.get("energy_budget_model", "unknown"))
                limit_mode_vals = nonthermal_props.get("inj_limit_mode_grid", np.array([]))
                limit_mode_vals = limit_mode_vals[nt_mask] if np.size(limit_mode_vals) > 0 else np.array([], dtype=int)

                if (
                    beta1_vals.size
                    and r1_vals.size
                    and theta_e1_vals.size
                    and theta_e2_ad_vals.size
                    and sironi_boost_vals.size
                    and theta_e_vals.size
                    and te2_vals.size
                    and gamma_failure_vals.size
                ):
                    dual_logger.human.log_stage_thermal_chain(
                        beta1=beta1_vals,
                        r1=r1_vals,
                        theta_e1=theta_e1_vals,
                        theta_e2_ad=theta_e2_ad_vals,
                        sironi_boost=sironi_boost_vals,
                        theta_e2=theta_e_vals,
                        te2=te2_vals,
                        gamma_min=gamma_vals,
                        gamma_failure=gamma_failure_vals,
                    )
                if sonic_vals.size and theta_bn_vals.size and inj_gate_vals.size and eta_vals.size and eps_vals.size:
                    dual_logger.human.log_stage_injection_gate(
                        sonic_mach=sonic_vals,
                        theta_bn=theta_bn_vals,
                        inj_gate=inj_gate_vals,
                        eta_inj_e=eta_vals,
                        eps_nth_e=eps_vals,
                        sigma_suppression=suppression_vals,
                    )
                if (
                    p_min_vals.size
                    and q_vals.size
                    and p_eff_vals.size
                    and c_vals.size
                    and n_nth_vals.size
                    and n_nth_eta_vals.size
                    and n_nth_eps_vals.size
                    and e_diss_e_vals.size
                    and e_diss_tot_vals.size
                    and u_nth_budget_vals.size
                    and gamma_vals.size
                    and limit_mode_vals.size
                ):
                    dual_logger.human.log_stage_radiation_interface(
                        p_min=p_min_vals,
                        q_vals=q_vals,
                        p_eff_vals=p_eff_vals,
                        unth_code=c_vals,
                        n_nth=n_nth_vals,
                        n_nth_eta=n_nth_eta_vals,
                        n_nth_eps=n_nth_eps_vals,
                        energy_budget_model=energy_budget_model,
                        e_diss_e=e_diss_e_vals,
                        e_diss_tot=e_diss_tot_vals,
                        u_nth_budget=u_nth_budget_vals,
                        gamma_min=gamma_vals,
                        limit_mode=limit_mode_vals,
                    )
                dual_logger.ai.data(
                    "stage_summary.radiation_interface",
                    {
                        "unth_code_median": float(np.median(c_vals)) if c_vals.size else 0.0,
                        "unth_code_min": float(np.min(c_vals)) if c_vals.size else 0.0,
                        "unth_code_max": float(np.max(c_vals)) if c_vals.size else 0.0,
                        "q_median": float(np.median(q_vals)) if q_vals.size else 0.0,
                        "p_classical_median": float(np.median(q_vals - 1.0)) if q_vals.size else 0.0,
                        "p_eff_median": float(np.median(p_eff_vals)) if p_eff_vals.size else 0.0,
                        "gamma_min_median": float(np.median(gamma_vals)) if gamma_vals.size else 1.0,
                        "gamma_min_min": float(np.min(gamma_vals)) if gamma_vals.size else 1.0,
                        "gamma_min_max": float(np.max(gamma_vals)) if gamma_vals.size else 1.0,
                        "gamma_valid_fraction": float(np.mean(gamma_failure_vals == 0)) if gamma_failure_vals.size else 1.0,
                        "gamma_fallback_risk_count": int(np.sum(gamma_vals <= 1.0)) if gamma_vals.size else 0,
                    },
                )

            sampling_stats = shock_props.get("sampling_stats")
            if sampling_stats:
                dual_logger.human.info(
                    "Shock sampling summary: "
                    f"verified={sampling_stats['verified_count']}, "
                    f"accepted_boundary_clipped={sampling_stats['boundary_clipped_verified_count']}"
                )
                dual_logger.human.info(
                    "SRMHD mainline summary: "
                    f"refined={sampling_stats.get('sr_refined_count', 0)}, "
                    f"rejected_low_mach={sampling_stats.get('sr_rejected_low_mach_count', 0)}, "
                    f"rejected_jump={sampling_stats.get('sr_rejected_jump_count', 0)}, "
                    f"rejected_entropy={sampling_stats.get('sr_rejected_entropy_count', 0)}"
                )

            dual_logger.ai.codepath(
                "Electron heating branch",
                "stage summaries only: shock acceptance, thermal chain, injection gate, radiation interface",
            )


        step_start = time.time()
        if dual_logger:
            dual_logger.ai.codepath("Step 4", "save_h5_file")
        save_h5_file(output_h5, roi_data, shock_props, nonthermal_props, config, dual_logger)
        stage_times["save"] = time.time() - step_start

        processing_time = time.time() - start_time
        if dual_logger:
            dual_logger.human.time("Save HDF5", stage_times["save"])
            dual_logger.human.log_finish(base_name, output_h5, processing_time)
            dual_logger.ai.func_exit(
                "process_snapshot",
                {"success": True, "output_h5": output_h5, "stage_times": stage_times},
            )
        elif logger:
            log_finish(logger, base_name, output_h5, processing_time)
        else:
            print(f"--- Finished {os.path.basename(output_h5)} ({processing_time:.2f}s) ---")
        return True

    except Exception as exc:
        processing_time = time.time() - start_time
        if dual_logger:
            dual_logger.human.log_error(base_name, str(exc))
            dual_logger.ai.exception(f"process_snapshot failed after {processing_time:.2f}s")
            dual_logger.ai.func_exit("process_snapshot", {"success": False, "error": str(exc)})
        elif logger:
            log_error(logger, base_name, str(exc), exc_info=True)
        else:
            print(f"--- Error processing {base_name}: {exc} ({processing_time:.2f}s) ---")
        return False
