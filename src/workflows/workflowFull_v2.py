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
        nt_params["rho_unit"] = rho_unit
        enable_advection = physics_cfg.get("enable_advection", False)

        if dual_logger:
            dual_logger.ai.debug(f"Derived units: L_unit={l_unit_val:.3e}, rho_unit={rho_unit:.3e}")
            dual_logger.ai.codepath("Physics branch", f"enable_advection={enable_advection}")

        shock_props, nonthermal_props = calculate_dsa_physics(
            roi_data,
            config["shock_params"],
            nt_params,
            dual_logger,
        )
        stage_times["dsa"] = time.time() - step_start

        if dual_logger:
            dual_logger.human.time("Shock detection + nonthermal electrons", stage_times["dsa"])

        if dual_logger and np.any(shock_props["mask"]):
            mask = shock_props["mask"]
            mach_vals = shock_props["upstream_mach"][mask]
            sigma_vals = shock_props.get("sigma_grid")
            sigma_vals = sigma_vals[mask] if sigma_vals is not None else None
            coverage = float(np.sum(mask) / mask.size)

            dual_logger.human.log_shock_stats(
                n_shock_cells=int(np.sum(mask)),
                mach_values=mach_vals,
                sigma_values=sigma_vals,
                coverage=coverage,
            )
            dual_logger.ai.data("shock_props.upstream_mach.active", mach_vals)
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
            mask = shock_props["mask"]
            c_grid = nonthermal_props.get("C_grid", np.array([]))
            q_grid = nonthermal_props.get("q_grid", np.array([]))
            gamma_min_grid = nonthermal_props.get("gamma_min_grid", np.array([]))
            sigma_suppression = nonthermal_props.get("sigma_suppression_grid")

            if np.size(c_grid) > 0:
                c_vals = c_grid[mask]
                q_vals = q_grid[mask]
                gamma_vals = gamma_min_grid[mask] if np.size(gamma_min_grid) > 0 else np.array([1.0])
                suppression_vals = sigma_suppression[mask] if sigma_suppression is not None else None

                dual_logger.human.log_nonthermal_stats(
                    n_inj_total=float(np.sum(c_vals)),
                    n_inj_range=(float(np.min(c_vals)), float(np.max(c_vals))),
                    p_values=q_vals,
                    gamma_min_values=gamma_vals,
                    sigma_suppression=suppression_vals,
                )
                dual_logger.ai.data("nonthermal.C_grid.active", c_vals)
                dual_logger.ai.data("nonthermal.q_grid.active", q_vals)
                dual_logger.ai.data("nonthermal.gamma_min.active", gamma_vals)

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
