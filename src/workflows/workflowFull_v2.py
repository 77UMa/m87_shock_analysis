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

            dual_logger.human.log_shock_stats(
                n_shock_cells=int(np.sum(mask)),
                mach_values=mach_vals,
                sigma_values=sigma_vals,
                coverage=coverage,
            )
            dual_logger.ai.data("shock_props.mainline_mach.active", mach_vals)
            if sigma_vals is not None:
                dual_logger.ai.data("shock_props.sigma.active", sigma_vals)
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
                gamma_vals = gamma_min_grid[nt_mask] if np.size(gamma_min_grid) > 0 else np.array([1.0])
                gamma_failure_vals = gamma_failure[nt_mask] if np.size(gamma_failure) > 0 else np.array([], dtype=int)
                suppression_vals = sigma_suppression[nt_mask] if sigma_suppression is not None else None

                dual_logger.human.log_nonthermal_stats(
                    n_inj_total=float(np.sum(c_vals)),
                    n_inj_range=(float(np.min(c_vals)), float(np.max(c_vals))),
                    p_values=q_vals,
                    gamma_min_values=gamma_vals,
                    sigma_suppression=suppression_vals,
                )
                dual_logger.human.info(
                    f"NT mainline shock cells: verified_candidates={int(np.sum(verified_mask))}; accepted_mainline={int(np.sum(nt_mask))}"
                )
                dual_logger.human.info(
                    "Branch summary: "
                    f"UNTH(code) median={np.median(c_vals):.3e}; gamma_min valid={(gamma_failure_vals == 0).sum()}/{gamma_failure_vals.size if gamma_failure_vals.size else 0}; "
                    f"fallback_risk={(gamma_vals <= 1.0).sum()}"
                )
                dual_logger.ai.data("nonthermal.unth_code.active", c_vals)
                dual_logger.ai.data("nonthermal.q_grid.active", q_vals)
                dual_logger.ai.data("nonthermal.gamma_min.active", gamma_vals)
                if gamma_failure_vals.size:
                    dual_logger.ai.data("nonthermal.gamma_min_failure.active", gamma_failure_vals)

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

            sigma2_grid = shock_props.get("sigma2_grid")
            if sigma2_grid is not None:
                dual_logger.ai.data("shock.sigma2.active", sigma2_grid[verified_mask])

            p2_over_rho2 = shock_props.get("press2_over_rho2_grid")
            if p2_over_rho2 is not None:
                dual_logger.ai.data("shock.press2_over_rho2.active", p2_over_rho2[verified_mask])

            sample_clipped = shock_props.get("sample_boundary_clipped_grid")
            if sample_clipped is not None:
                dual_logger.ai.data("shock.sample_boundary_clipped.active", sample_clipped[verified_mask])

            theta_e = nonthermal_props.get("theta_e_grid")
            if theta_e is not None and np.size(theta_e) > 0:
                dual_logger.ai.data("nonthermal.theta_e.active", theta_e[nt_mask])
            theta_e1 = nonthermal_props.get("theta_e1_grid")
            if theta_e1 is not None and np.size(theta_e1) > 0:
                dual_logger.ai.data("nonthermal.theta_e1.active", theta_e1[nt_mask])
            theta_e2_ad = nonthermal_props.get("theta_e2_ad_grid")
            if theta_e2_ad is not None and np.size(theta_e2_ad) > 0:
                dual_logger.ai.data("nonthermal.theta_e2_ad.active", theta_e2_ad[nt_mask])
            sironi_boost = nonthermal_props.get("sironi_boost_grid")
            if sironi_boost is not None and np.size(sironi_boost) > 0:
                dual_logger.ai.data("nonthermal.sironi_boost.active", sironi_boost[nt_mask])

            p_min_phys = nonthermal_props.get("p_min_physical_grid")
            if p_min_phys is not None and np.size(p_min_phys) > 0:
                dual_logger.ai.data("nonthermal.p_min_physical.active", p_min_phys[nt_mask])
            eta_inj_e = nonthermal_props.get("eta_inj_e_grid")
            if eta_inj_e is not None and np.size(eta_inj_e) > 0:
                dual_logger.ai.data("nonthermal.eta_inj_e.active", eta_inj_e[nt_mask])
            eps_nth_e = nonthermal_props.get("eps_nth_e_grid")
            if eps_nth_e is not None and np.size(eps_nth_e) > 0:
                dual_logger.ai.data("nonthermal.eps_nth_e.active", eps_nth_e[nt_mask])
            inj_gate = nonthermal_props.get("inj_gate_grid")
            if inj_gate is not None and np.size(inj_gate) > 0:
                dual_logger.ai.data("nonthermal.inj_gate.active", inj_gate[nt_mask])
            n_nth_phys = nonthermal_props.get("n_nth_phys_grid")
            if n_nth_phys is not None and np.size(n_nth_phys) > 0:
                dual_logger.ai.data("nonthermal.n_nth_phys.active", n_nth_phys[nt_mask], "cm^-3")

            dual_logger.ai.codepath(
                "Electron heating branch",
                "Single two-temperature/Sironi-Tran heating plus PIC dual-cap injection gating",
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
