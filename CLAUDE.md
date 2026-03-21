# ⚠️ CRITICAL: Token & Performance Policy

### 1. Think-Before-Acting Protocol (CRITICAL)
- **NO PROACTIVE SCANNING**: 严禁在未经过用户确认前对整个项目进行大规模 Hashing 或文件读取。
- **DISCUSSION FIRST**: 在执行任何涉及多于 3 个文件的读取、编辑或运行复杂 shell 命令之前，必须先向用户简报你的“分析逻辑”和“预想步骤”。
- **TOKEN QUOTA AWARENESS**: 意识到 Token 消耗成本。如果任务涉及大数据文件（.hdf5, .dat）或大型子模块（ipole-DSA），优先询问用户是否可以跳过。

### 2. Context Awareness & Memory
- **VERIFY STATE**: 每次任务开始时，先运行 `git log -n 1` 确认当前所处的真实 Git 分支和最后提交时间，严禁产生“虚假提交”或“记混历史”的幻觉。
- **SUBMODULE POLICY**: 除非明确要求修改 C 代码，否则严禁扫描 `ipole-DSA/` 内部文件。将其视为黑盒调用。

### 3. Execution Rules
- **LOCAL VS REMOTE**: 区分本地开发环境和远程服务器环境。在执行编译（make）或大数据处理前，必须确认当前环境的计算资源。
- **NO SILENT HANG**: 如果某个内部步骤（如索引）预计超过 30 秒，必须立即告知用户原因，并提供中断选项。

# Project Info
## 1. Project Context: M87 Jet DSA Model

- **Scientific Goal**: Test if Diffusive Shock Acceleration (DSA) can explain M87 jet limb-brightening, as an alternative to the Magnetic Reconnection model (Yang et al. 2024).
    
- **Core Physics**:
    
    - Shock detection via normal Mach number (Lovely & Haimes 1999) and entropy jump.
        
    - Steady-State Advection-Cooling Approximation for $N_{nth}$ distribution.
        
    - $\sigma$-suppression efficiency: $\xi_{\rm DSA}(\sigma) = \xi_0 \cdot [1 + (\sigma/\sigma_{\rm crit})^\alpha]^{-1}$.
        
- **Primary Branch**: `feature/sigma-suppression-methodB` (Active development for $\sigma$ effects).
    

---

## 2. Technical Stack & Workflow

- **Data Source**: Athena++ output (`.athdf`).
    
- **Processing**: Python pipeline (`run_dsa_pipeline.py`, `run_sigma_sweep.py`).
    
- **Radiative Transfer**: `ipole-DSA` (Customized C-code for $C$ and $p$ mapping).
    
- **Deployment**:
    
    - Development: Local machine (via Claude Code).
        
    - Execution/Sim: Remote Server (Sync via GitHub: `77UMa/m87_shock_analysis`).

## 3. Communication & Memory Protocol

- **Session Continuity**:
    
    - Refer to `项目状态更新：M87喷流的激波加速模型验证.md` for the latest scientific baseline.
        
    - If a task involves multiple steps, list the plan as a checklist and wait for user "GO".
        
- **Physical Constants**: Ensure all unit conversions between Athena++ (code units) and IPOLE (physical units) are cross-checked with `Xia et al. (2025)`.
    

---

## 4. File Structure Shortcuts

- **Pipeline Logic**: `workflowFull_v2.py`
    
- **$\sigma$-Suppression Logic**: `nt_electron_v1.py` & `advection_v0.py`
    
- **IPOLE Input Gen**: `compare_models_v0.py` (HDF5 structure)
    
- **External**: `pyathena` (for reading `.athdf`)
