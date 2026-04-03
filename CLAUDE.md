项目技术手册可以参阅/docs 其中最重要的文件是不断维护更新中的engineering_guide.md
主模块（本文件夹）通过Mutagen双向传输对应服务器文件夹/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238/
Projects/MAD98_DSA_Postprocessing ，并对应github项目 https://github.com/77UMa/m87_shock_analysis ，目前正处于分支 feature/sigma-suppression-methodB 的开发中
子模块 ./ipole-DSA 对应github项目 https://github.com/77UMa/ipole ，目前也处于分支 feature/sigma-suppression-methodB 的开发中

# ⚠️ CRITICAL: Methodology Computational Physics&SWE
区别通用软件工程后端开发和我们正在进行的计算物理项目的方法论：
我正在开发计算天体物理代码，在引入/修正物理时需要区分两种情况。
## 第一性原理
第一性原理错误造成的bug属于对大自然的fundamental law的违反，而非企业客户选择A or B方案的体验差距。比如使用经典MHD算法探测了黑洞相对论性喷流中的激波。此时请收起前端开发和微服务架构里的 A/B 测试、容错降级和并行保留旧逻辑的习惯。
1. 追求物理自洽，如果物理量不合法，必须 Fail Loudly（大声报错）。
2. 若我们讨论确认了新物理公式的合理性，对于新公式直接替换旧公式，我会用 Git 控制版本，不要在代码里写新旧双链。
3. 专注于性能和物理正确性。
## 唯象模型/经验公式
这类比如我们要引入次网格模型的修正。我们此时的目标不是不是“确立真理”，而是“探索这种效应会如何影响最终的可观测结果”。
这时可以借鉴通用软件工程，使用模块化隔离，配置驱动探索的方法。

### 1. Think-Before-Acting Protocol 
- **NO PROACTIVE SCANNING**: 严禁在未经过用户确认前对整个项目进行大规模 Hashing 或文件读取。

### 2. Context Awareness & Memory
- **SUBMODULE POLICY**: 除非明确了怀疑对象在ipole的C端，否则不要扫描 `ipole-DSA/` 内部文件。将其视为黑盒调用。

编辑器崩溃预防: 在使用Diff 补丁模式查找长Python文件信息时，编辑器可能会崩溃，插入大段的空行并对程序产生语义污染。当你评估可能发生或者已经发生这种事情的时候，请立即停止并使用简短的语言描述你的整个调整计划。

### 3. Execution Rules
- **LOCAL VS REMOTE**: 区分本地开发环境和远程服务器环境，这意味着不需要跑复杂验证，通过编译测试即可。


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
