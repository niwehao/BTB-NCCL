#!/bin/bash
# SimAI htsim 一站式运行脚本
# 读取拓扑配置 + workload 配置, 生成 workload (如需), 编译, 运行仿真, 保存配置到 log 目录
#
# 用法:
#   ./run.sh conf/topo/fattree.json conf/workload/deepseek-671b-decode.json
#   ./run.sh conf/topo/fattree.json conf/topo/mixnet.json conf/workload/deepseek-671b-decode.json
#   ./run.sh conf/topo/fattree.json conf/workload/deepseek-671b-decode.json --skip-build
#   ./run.sh --build-only
#   ./run.sh --test-all [workload.json]

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="${ROOT}/simai_htsim"

# ========== 解析参数 ==========
TOPO_CFGS=()
WORKLOAD_CFG=""
BUILD=true
SKIP_BUILD=false
BUILD_ONLY=false
TEST_ALL=false
GEN_WORKLOAD=false
BUILD_JOBS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only)     BUILD_ONLY=true; shift ;;
    --skip-build)     SKIP_BUILD=true; shift ;;
    --test-all)       TEST_ALL=true; shift ;;
    --gen-workload)   GEN_WORKLOAD=true; shift ;;
    -j)               BUILD_JOBS="$2"; shift 2 ;;
    */topo/*.json)    TOPO_CFGS+=("$1"); shift ;;
    */workload/*.json) WORKLOAD_CFG="$1"; shift ;;
    *.json)
      # 自动判断: 含 topology 字段的是拓扑配置, 含 model 字段的是 workload 配置
      if jq -e '.topology' "$1" &>/dev/null; then
        TOPO_CFGS+=("$1")
      elif jq -e '.model' "$1" &>/dev/null; then
        WORKLOAD_CFG="$1"
      else
        echo "Error: cannot determine type of $1 (no 'topology' or 'model' key)"
        exit 1
      fi
      shift ;;
    *)  echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ========== 编译 ==========
do_build() {
  echo "====== Building simai_htsim (j=$BUILD_JOBS) ======"
  bash "${ROOT}/build.sh" -j "$BUILD_JOBS"
  echo ""
}

if $BUILD_ONLY; then
  do_build
  echo "Build complete: $BIN"
  exit 0
fi

if ! $SKIP_BUILD; then
  do_build
fi

if [[ ! -f "$BIN" ]]; then
  echo "Error: $BIN not found. Run without --skip-build first."
  exit 1
fi

# ========== 依赖检查 ==========
if ! command -v jq &>/dev/null; then
  echo "Error: jq is required. Install: brew install jq"
  exit 1
fi

# ========== 读取两个 JSON 配置并运行仿真的函数 ==========
run_config() {
  local topo_cfg="$1"
  local wl_cfg="$2"

  if [[ ! -f "$topo_cfg" ]]; then
    echo "Error: topo config not found: $topo_cfg"
    return 1
  fi
  if [[ ! -f "$wl_cfg" ]]; then
    echo "Error: workload config not found: $wl_cfg"
    return 1
  fi

  echo "====== Topo: $topo_cfg | Workload: $wl_cfg ======"

  # --- 读取 model (from workload config) ---
  local WORKLOAD=$(jq -r '.model.workload // ""' "$wl_cfg")
  local MODEL_NAME=$(jq -r '.model.name // "unknown"' "$wl_cfg")
  local MODEL_SIZE=$(jq -r '.model.model_size // ""' "$wl_cfg")
  local WORLD_SIZE=$(jq -r '.model.world_size // 64' "$wl_cfg")
  local TP=$(jq -r '.model.tp_degree // 1' "$wl_cfg")
  local PP=$(jq -r '.model.pp_degree // 1' "$wl_cfg")
  local DP=$(jq -r '.model.dp_degree // 1' "$wl_cfg")
  local EP=$(jq -r '.model.ep_degree // 1' "$wl_cfg")
  local GPUS_PER_SERVER=$(jq -r '.model.gpus_per_server // 8' "$wl_cfg")
  local PHASE=$(jq -r '.model.phase // "decode"' "$wl_cfg")
  local SEQ_LENGTH=$(jq -r '.model.seq_length // 1024' "$wl_cfg")
  local MICRO_BATCH=$(jq -r '.model.micro_batch // 1' "$wl_cfg")

  # --- 读取 topology (from topo config) ---
  local TOPO=$(jq -r '.topology.type // "fattree"' "$topo_cfg")
  local SPEED=$(jq -r '.topology.speed // 100000' "$topo_cfg")
  local QUEUESIZE=$(jq -r '.topology.queuesize // 8' "$topo_cfg")
  local ALPHA=$(jq -r '.topology.alpha // 4' "$topo_cfg")
  local RECONF_DELAY=$(jq -r '.topology.reconf_delay // 10' "$topo_cfg")
  local ECS_ONLY=$(jq -r '.topology.ecs_only // false' "$topo_cfg")
  local OS_RATIO=$(jq -r '.topology.os_ratio // 2' "$topo_cfg")

  # --- 读取 simulation (from topo config) ---
  local ITERATIONS=$(jq -r '.simulation.iterations // 1' "$topo_cfg")

  # --- workload 路径自动推导 ---
  if [[ -z "$WORKLOAD" || "$WORKLOAD" == "null" ]]; then
    WORKLOAD="SimAI/aicb/results/workload/${MODEL_NAME}-tp${TP}-pp${PP}-ep${EP}-dp${DP}-${WORLD_SIZE}gpu-${PHASE}.txt"
    echo "  Workload path (auto): $WORKLOAD"
  fi

  # --- workload 生成 (如需) ---
  if [[ ! -f "$WORKLOAD" ]] || $GEN_WORKLOAD; then
    echo "  Generating workload..."
    if [[ -z "$MODEL_SIZE" || "$MODEL_SIZE" == "null" ]]; then
      echo "  Error: model.model_size required for workload generation"
      return 1
    fi
    (
      cd "${ROOT}/SimAI/aicb"
      bash scripts/inference_workload_with_aiob.sh \
        --model_size "$MODEL_SIZE" \
        --phase "$PHASE" \
        --seq_length "$SEQ_LENGTH" \
        --micro_batch "$MICRO_BATCH" \
        --world_size "$WORLD_SIZE" \
        --tensor_model_parallel_size "$TP" \
        --pipeline_model_parallel "$PP" \
        --expert_model_parallel_size "$EP" \
        --result_dir "results/workload/"
    )
    if [[ ! -f "$WORKLOAD" ]]; then
      echo "  Error: workload generation did not produce: $WORKLOAD"
      echo "  Check SimAI/aicb/results/workload/ for generated files"
      ls SimAI/aicb/results/workload/ 2>/dev/null | tail -5
      return 1
    fi
    echo "  Workload generated: $WORKLOAD"
  fi

  # --- 构建 simai_htsim 命令 ---
  local CMD=("$BIN")
  CMD+=(--workload "$WORKLOAD")
  CMD+=(--topo "$TOPO")
  CMD+=(--nodes "$WORLD_SIZE")
  CMD+=(--gpus_per_server "$GPUS_PER_SERVER")
  CMD+=(--dp_degree "$DP")
  CMD+=(--tp_degree "$TP")
  CMD+=(--pp_degree "$PP")
  CMD+=(--ep_degree "$EP")
  CMD+=(--speed "$SPEED")
  CMD+=(--queuesize "$QUEUESIZE")
  CMD+=(--iterations "$ITERATIONS")

  # 拓扑特有参数
  case "$TOPO" in
    mixnet)
      CMD+=(--alpha "$ALPHA")
      CMD+=(--reconf_delay "$RECONF_DELAY")
      if [[ "$ECS_ONLY" == "true" ]]; then
        CMD+=(--ecs_only)
      fi
      ;;
    os_fattree|agg_os_fattree)
      CMD+=(--os_ratio "$OS_RATIO")
      ;;
  esac

  echo "  Topo: $TOPO | GPUs: $WORLD_SIZE | Speed: ${SPEED}Mbps | Iter: $ITERATIONS"
  echo "  CMD: ${CMD[*]}"
  echo ""

  # --- 运行仿真 ---
  local STDERR_TMP=$(mktemp)
  "${CMD[@]}" 2> >(tee "$STDERR_TMP" >&2)
  local EXIT_CODE=$?

  # --- 后处理: 复制配置到 log 目录 ---
  local LOG_DIR=$(grep '\[LOG\] Output directory:' "$STDERR_TMP" 2>/dev/null | awk '{print $NF}')
  rm -f "$STDERR_TMP"

  if [[ -n "$LOG_DIR" && -d "$LOG_DIR" ]]; then
    cp "$topo_cfg" "$LOG_DIR/topo.json"
    cp "$wl_cfg" "$LOG_DIR/workload.json"
    # 移动 ncclFlowModel_EndToEnd.csv 到 log 目录
    if [[ -f "${ROOT}/ncclFlowModel_EndToEnd.csv" ]]; then
      mv "${ROOT}/ncclFlowModel_EndToEnd.csv" "$LOG_DIR/"
    fi
    echo ""
    echo "  Configs saved to: $LOG_DIR/topo.json, $LOG_DIR/workload.json"
    echo "  Log directory: $LOG_DIR"
  fi

  return $EXIT_CODE
}

# ========== 批量运行多个拓扑 ==========
run_multiple() {
  local wl_cfg="$1"
  shift
  local topo_list=("$@")

  echo "====== Batch run: ${#topo_list[@]} topologies ======"
  echo "  Workload: $wl_cfg"
  echo ""

  local PASS=0
  local FAIL=0
  for cfg in "${topo_list[@]}"; do
    local TOPO_NAME=$(jq -r '.topology.type' "$cfg")
    echo ">>> [$TOPO_NAME] $cfg"
    if run_config "$cfg" "$wl_cfg"; then
      echo "<<< [$TOPO_NAME] OK"
      ((PASS++))
    else
      echo "<<< [$TOPO_NAME] FAILED (exit $?)"
      ((FAIL++))
    fi
    echo ""
  done

  echo "====== Results: $PASS passed, $FAIL failed ======"
  return $FAIL
}

# ========== test-all: 用所有拓扑配置逐一测试 ==========
if $TEST_ALL; then
  TOPO_CFGS=(${ROOT}/conf/topo/*.json)
  # 如果未指定 workload 配置, 使用第一个
  if [[ -z "$WORKLOAD_CFG" ]]; then
    WORKLOAD_CFG=$(ls ${ROOT}/conf/workload/*.json 2>/dev/null | head -1)
  fi

  if [[ ${#TOPO_CFGS[@]} -eq 0 ]]; then
    echo "Error: no topo configs found in conf/topo/"
    exit 1
  fi
  if [[ -z "$WORKLOAD_CFG" ]]; then
    echo "Error: no workload configs found in conf/workload/"
    exit 1
  fi

  run_multiple "$WORKLOAD_CFG" "${TOPO_CFGS[@]}"
  exit $?
fi

# ========== 单次或多拓扑运行 ==========
if [[ ${#TOPO_CFGS[@]} -eq 0 || -z "$WORKLOAD_CFG" ]]; then
  echo "Usage:"
  echo "  $0 <topo.json> <workload.json>                          # 单拓扑运行"
  echo "  $0 <topo1.json> <topo2.json> ... <workload.json>        # 多拓扑批量运行"
  echo "  $0 <topo.json> <workload.json> --skip-build             # 跳过编译"
  echo "  $0 --build-only                                          # 仅编译"
  echo "  $0 --test-all [workload.json]                            # 测试所有拓扑"
  echo ""
  echo "Topo configs:"
  ls conf/topo/*.json 2>/dev/null | sed 's/^/  /'
  echo ""
  echo "Workload configs:"
  ls conf/workload/*.json 2>/dev/null | sed 's/^/  /'
  exit 1
fi

if [[ ${#TOPO_CFGS[@]} -eq 1 ]]; then
  run_config "${TOPO_CFGS[0]}" "$WORKLOAD_CFG"
else
  run_multiple "$WORKLOAD_CFG" "${TOPO_CFGS[@]}"
fi
