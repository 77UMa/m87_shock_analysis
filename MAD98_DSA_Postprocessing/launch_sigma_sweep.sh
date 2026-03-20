#!/bin/bash
# launch_sigma_sweep.sh
#
# 在 screen 离线会话中顺序运行 sigma 扫描实验，捕获全部终端输出到日志。
#
# 用法:
#   bash launch_sigma_sweep.sh <snapshot.athdf> [output_base_dir]
#
# 示例:
#   bash launch_sigma_sweep.sh \
#       /cpfs01/.../data/mad98.prim.00469.athdf \
#       /cpfs01/.../sigma_sweep
#
# 日志说明:
#   sweep_*.log     — Python logger 输出（结构化，含时间戳）
#   screen_*.log    — tee 捕获的全部终端输出（含 ipole verbose 打印）
#
# 查看运行状态:
#   screen -r <session_name>      # 接入 screen 实时查看
#   tail -f <output_dir>/screen_*.log    # 不接入 screen 直接追踪日志

set -euo pipefail

# ── 参数解析 ──────────────────────────────────────────────────────────────────
SNAPSHOT="${1:-}"
if [ -z "${SNAPSHOT}" ]; then
    echo "Usage: bash $0 <snapshot.athdf> [output_base_dir]"
    echo ""
    echo "Example:"
    echo "  bash $0 /cpfs01/.../data/mad98.prim.00469.athdf"
    exit 1
fi

SNAP_BASE=$(basename "${SNAPSHOT%.athdf}")
CPFS_DEFAULT="/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238/DSA/sigma_sweep"
OUTPUT_BASE="${2:-${CPFS_DEFAULT}}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${SNAP_BASE}_${TIMESTAMP}"

# screen 会话名：截短以防超长
SESSION="sweep_${SNAP_BASE:0:20}_${TIMESTAMP:8}"   # 只取时间后6位

# screen_*.log 存在 output_dir 下，方便与 Python 日志放一起
SCREEN_LOG="${OUTPUT_DIR}/screen_${SNAP_BASE}_${TIMESTAMP}.log"

# ── 脚本所在目录（保证路径正确）────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_PY="${SCRIPT_DIR}/MAD98_DSA_Postprocessing/run_sigma_sweep.py"

if [ ! -f "${SWEEP_PY}" ]; then
    echo "Error: run_sigma_sweep.py not found at ${SWEEP_PY}"
    exit 1
fi

# ── 创建输出目录并写入启动记录 ───────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/launch_info.txt" << EOF
Sigma Sweep Launch Record
=========================
Snapshot   : ${SNAPSHOT}
Sigma vals : 0.01  0.03  0.10
Alpha      : 2
Output dir : ${OUTPUT_DIR}
Session    : ${SESSION}
Screen log : ${SCREEN_LOG}
Launched   : $(date)
Host       : $(hostname)
EOF

# ── 打印启动信息 ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Launching sigma sweep in screen"
echo "  Session    : ${SESSION}"
echo "  Snapshot   : ${SNAPSHOT}"
echo "  Sigma vals : 0.01  0.03  0.10"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Screen log : ${SCREEN_LOG}"
echo "============================================================"
echo ""
echo "To attach session : screen -r ${SESSION}"
echo "To tail log       : tail -f ${SCREEN_LOG}"
echo ""

# ── 启动 screen 离线会话 ──────────────────────────────────────────────────────
# 说明：
#   - screen -dmS <name>: 后台创建新会话，不立即接入
#   - bash -c "...": 在会话中执行命令串
#   - 2>&1 | tee: 同时将 stdout+stderr 写入 screen_*.log
#   - exec bash: 任务结束后保持 screen 存活（方便事后检查输出）
screen -dmS "${SESSION}" bash -c "
    cd '${SCRIPT_DIR}'

    # ── 标记开始 ────────────────────────────────────────────────────────────
    echo '============================================================'
    echo 'SIGMA SWEEP SESSION STARTED'
    echo \"Snapshot : ${SNAPSHOT}\"
    echo \"Output   : ${OUTPUT_DIR}\"
    echo \"Started  : \$(date)\"
    echo '============================================================'

    # ── 运行 Python 脚本，tee 捕获全部输出 ──────────────────────────────────
    python '${SWEEP_PY}' \
        --snapshot '${SNAPSHOT}' \
        --sigma-values 0.01 0.03 0.1 \
        --alpha-sigma 2 \
        --output-dir '${OUTPUT_DIR}' \
        2>&1 | tee '${SCREEN_LOG}'

    EXIT_CODE=\${PIPESTATUS[0]}

    # ── 标记结束 ────────────────────────────────────────────────────────────
    echo ''
    echo '============================================================'
    if [ \${EXIT_CODE} -eq 0 ]; then
        echo 'SIGMA SWEEP FINISHED SUCCESSFULLY'
    else
        echo \"SIGMA SWEEP FAILED (exit code: \${EXIT_CODE})\"
    fi
    echo \"Finished : \$(date)\"
    echo \"Log      : ${SCREEN_LOG}\"
    echo '============================================================'

    # 保持 session 存活，方便事后 attach 检查
    exec bash
"

# ── 确认 screen 已启动 ────────────────────────────────────────────────────────
sleep 0.5
if screen -list | grep -q "${SESSION}"; then
    echo "Screen session '${SESSION}' is running."
    echo ""
    echo "Quick commands:"
    echo "  Attach     : screen -r ${SESSION}"
    echo "  Detach     : Ctrl+A, D"
    echo "  Tail log   : tail -f ${SCREEN_LOG}"
    echo "  Check done : grep 'SWEEP FINISHED\|SWEEP FAILED' ${SCREEN_LOG}"
else
    echo "Warning: screen session may not have started correctly."
    echo "Check with: screen -ls"
fi
