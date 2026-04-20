#!/bin/bash
# 批量跑 conf/workload_tmp/ 里的所有 workload JSON × 指定拓扑。
# 支持并行:默认 4 个任务并发。
# 每个组合产出一条 log,所有产物(per-run log 目录 + TSV 汇总)放进一个批次目录,
# 不污染 ./log/。
#
# 用法:
#   ./run_batch_tmp.sh                         # 默认 mixnet + fattree,parallel=4
#   ./run_batch_tmp.sh conf/topo/mixnet.json   # 只跑指定拓扑
#   ./run_batch_tmp.sh --topos=mixnet,fattree  # 多拓扑逗号分隔
#   ./run_batch_tmp.sh --gen                   # 强制重新生成 workload txt
#   ./run_batch_tmp.sh --build                 # 先编译再跑
#   ./run_batch_tmp.sh --parallel=8            # 并行度(默认 4)
#   ./run_batch_tmp.sh -P 2                    # 同上,短格式
#   ./run_batch_tmp.sh --out=path/to/dir       # 指定批次输出根目录

set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
TMP_DIR="${ROOT}/conf/workload_tmp"
DEFAULT_TOPOS=("${ROOT}/conf/topo/mixnet.json" "${ROOT}/conf/topo/fattree.json")
DEFAULT_OUT_ROOT="${ROOT}/log_batch_tmp"

TOPOS=()
DO_BUILD=false
DO_GEN=false
OUT_DIR=""
PARALLEL=2

# 参数解析 (支持 --parallel=N / -P N / --parallel N)
i=1
args=("$@")
while (( i <= $# )); do
  arg="${args[$((i-1))]}"
  case "$arg" in
    --build) DO_BUILD=true ;;
    --gen|--gen-workload) DO_GEN=true ;;
    --topos=*) IFS=',' read -ra LIST <<< "${arg#--topos=}"
               for t in "${LIST[@]}"; do TOPOS+=("${ROOT}/conf/topo/${t%.json}.json"); done ;;
    --out=*)   OUT_DIR="${arg#--out=}" ;;
    --parallel=*) PARALLEL="${arg#--parallel=}" ;;
    -P) i=$((i+1)); PARALLEL="${args[$((i-1))]}" ;;
    -j) i=$((i+1)); PARALLEL="${args[$((i-1))]}" ;;
    *.json)    TOPOS+=("$arg") ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
  i=$((i+1))
done
if [[ ${#TOPOS[@]} -eq 0 ]]; then
  TOPOS=("${DEFAULT_TOPOS[@]}")
fi
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="${DEFAULT_OUT_ROOT}/batch_$(date +%y%m%d_%H%M%S)"
fi
mkdir -p "$OUT_DIR" "$OUT_DIR/runs" "$OUT_DIR/rows"
[[ $PARALLEL -lt 1 ]] && PARALLEL=1

echo "=============================================="
echo "Batch runner (parallel=$PARALLEL)"
echo "  tmp dir   : $TMP_DIR"
echo "  topos     : ${TOPOS[*]}"
echo "  gen wl    : $DO_GEN"
echo "  build     : $DO_BUILD"
echo "  out dir   : $OUT_DIR"
echo "=============================================="

if $DO_BUILD; then
  echo "[BUILD] ./build.sh -j 8"
  bash "${ROOT}/build.sh" -j 8 || { echo "BUILD FAILED"; exit 1; }
fi

# 枚举 tmp 下的所有 JSON (按文件名排序)
WLS=()
while IFS= read -r f; do WLS+=("$f"); done < <(ls -1 "$TMP_DIR"/*.json 2>/dev/null | sort)

if [[ ${#WLS[@]} -eq 0 ]]; then
  echo "No workload JSON found under $TMP_DIR"
  exit 1
fi

# 单个任务的执行函数 (在 & 后台里调用)
run_one() {
  local idx="$1" topo="$2" wl="$3"
  local WL_NAME TOPO_NAME OUT_FILE ROW_FILE
  WL_NAME=$(basename "$wl" .json)
  TOPO_NAME=$(basename "$topo" .json)
  OUT_FILE="$OUT_DIR/runs/${idx}_${WL_NAME}__${TOPO_NAME}.out"
  ROW_FILE="$OUT_DIR/rows/${idx}.tsv"

  echo "[START idx=$idx] $TOPO_NAME × $WL_NAME"

  local ARGS=("$topo" "$wl" --skip-build)
  $DO_GEN && ARGS+=(--gen-workload)
  bash "${ROOT}/run.sh" "${ARGS[@]}" >"$OUT_FILE" 2>&1
  local EC=$?

  if [[ $EC -ne 0 ]]; then
    echo "[DONE  idx=$idx] FAIL ($TOPO_NAME × $WL_NAME, exit=$EC)"
    printf "%s\t%s\tERR\t-\t-\t-\t-\t-\t%s\n" \
      "$TOPO_NAME" "$WL_NAME" "$OUT_FILE" > "$ROW_FILE"
    return
  fi

  # 从 run.sh 的 stdout 里抓 "Log directory:"(并行下比时间戳扫描可靠)
  local SRC_LOG
  SRC_LOG=$(grep "Log directory:" "$OUT_FILE" | tail -1 | awk '{print $NF}')
  local LOG_DIR=""
  if [[ -n "$SRC_LOG" && -d "$SRC_LOG" ]]; then
    local DST_LOG="$OUT_DIR/${WL_NAME}__$(basename "$SRC_LOG")"
    mv "$SRC_LOG" "$DST_LOG" 2>/dev/null && LOG_DIR="$DST_LOG"
  fi

  if [[ -z "$LOG_DIR" || ! -f "$LOG_DIR/stats.txt" ]]; then
    echo "[DONE  idx=$idx] NO_STATS ($TOPO_NAME × $WL_NAME, src=$SRC_LOG)"
    printf "%s\t%s\tNO_STATS\t-\t-\t-\t-\t-\t%s\n" \
      "$TOPO_NAME" "$WL_NAME" "${LOG_DIR:-$SRC_LOG}" > "$ROW_FILE"
    return
  fi

  local SIM P1 PAVG RCT RSK RETX
  SIM=$(grep "Final sim time"     "$LOG_DIR/stats.txt" | awk '{print $4}')
  P1=$(grep  "^Pass 1:"            "$LOG_DIR/stats.txt" | awk '{print $3}')
  PAVG=$(grep "Avg per pass"       "$LOG_DIR/stats.txt" | awk '{print $4}')
  RCT=$(grep  "Reconfigs triggered" "$LOG_DIR/stats.txt" | awk '{print $NF}')
  RSK=$(grep  "Reconfigs skipped"  "$LOG_DIR/stats.txt" | awk '{print $NF}')
  RETX=$(grep "Total retransmissions" "$LOG_DIR/stats.txt" | awk '{print $NF}')

  echo "[DONE  idx=$idx] $TOPO_NAME × $WL_NAME  sim=${SIM:-?}ms pass_avg=${PAVG:-?}ms reconf=${RCT:-0}/+${RSK:-0}skip retx=${RETX:-?}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$TOPO_NAME" "$WL_NAME" "${SIM:-?}" "${P1:-?}" "${PAVG:-?}" \
    "${RCT:-0}" "${RSK:-0}" "${RETX:-?}" "$LOG_DIR" > "$ROW_FILE"
}

# 调度:维持 ≤ PARALLEL 个并发,wait -n 释放槽位
TOTAL=$(( ${#WLS[@]} * ${#TOPOS[@]} ))
echo "Total tasks: $TOTAL | concurrency: $PARALLEL"
echo ""

I=0
active=0
for wl in "${WLS[@]}"; do
  for topo in "${TOPOS[@]}"; do
    I=$((I+1))
    run_one "$(printf "%03d" "$I")" "$topo" "$wl" &
    active=$((active+1))
    # 满员就等一个退出
    while (( active >= PARALLEL )); do
      wait -n
      active=$((active-1))
    done
  done
done

# 等剩余的
wait

# 拼总表
SUMMARY="${OUT_DIR}/summary.tsv"
echo -e "topo\tworkload\tsim_time_ms\tpass1_ms\tpass_avg_ms\treconfigs_trig\treconfigs_skip\tretx\tfinal_log" > "$SUMMARY"
for f in $(ls -1 "$OUT_DIR/rows"/*.tsv 2>/dev/null | sort); do
  cat "$f" >> "$SUMMARY"
done

echo ""
echo "=============================================="
echo "Batch finished. Summary: $SUMMARY"
echo "=============================================="
column -s $'\t' -t "$SUMMARY"
