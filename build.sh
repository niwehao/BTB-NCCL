#!/bin/bash
# 全量编译 simai_htsim
# 用法:
#   ./build.sh          全量编译
#   ./build.sh -j 8     并行编译 (8线程)

set -e

# 自动定位项目根目录 (build.sh 所在目录)
ROOT="$(cd "$(dirname "$0")" && pwd)"
HTSIM_SRC="${ROOT}/mixnet-sim/mixnet-htsim/src/clos"
SIMAI_SRC="${ROOT}/SimAI/astra-sim-alibabacloud"
BUILD="${ROOT}/build_htsim"
OUT="${ROOT}/simai_htsim"

# 自动查找 flatbuffers include 路径
if [ -d "$HOME/.local/include/flatbuffers" ]; then
    FLATBUF_INC="$HOME/.local/include"
elif [ -d "/opt/homebrew/Cellar/flatbuffers" ]; then
    FLATBUF_INC="$(find /opt/homebrew/Cellar/flatbuffers -maxdepth 2 -name include -type d | head -1)"
elif [ -d "/usr/local/include/flatbuffers" ]; then
    FLATBUF_INC="/usr/local/include"
elif [ -d "/usr/include/flatbuffers" ]; then
    FLATBUF_INC="/usr/include"
else
    echo "Error: flatbuffers not found"
    exit 1
fi

CXXFLAGS="-Wall -std=c++17 -O0 -g -Wno-deprecated"
INCLUDES="-I${HTSIM_SRC} -I${HTSIM_SRC}/datacenter -I${FLATBUF_INC} -I${SIMAI_SRC}"

# 解析参数
JOBS=1
while getopts "j:" opt; do
    case $opt in
        j) JOBS=$OPTARG ;;
        *) echo "Usage: $0 [-j N]"; exit 1 ;;
    esac
done

echo "ROOT:    ${ROOT}"
echo "FLATBUF: ${FLATBUF_INC}"

# ======== 源文件白名单 (只编译需要的文件) ========

HTSIM_FILES="
cbr.cpp
cbrpacket.cpp
clock.cpp
compositeprioqueue.cpp
compositequeue.cpp
config.cpp
cpqueue.cpp
dctcp.cpp
dyn_net_sch.cpp
ecnqueue.cpp
eth_pause_packet.cpp
eventlist.cpp
exoqueue.cpp
fairpullqueue.cpp
ffapp.cpp
logfile.cpp
loggers.cpp
mixnet_topomanager.cpp
mtcp.cpp
ndp.cpp
ndppacket.cpp
network.cpp
pipe.cpp
prioqueue.cpp
qcn.cpp
queue.cpp
queue_lossless.cpp
queue_lossless_input.cpp
queue_lossless_output.cpp
randomqueue.cpp
route.cpp
sent_packets.cpp
switch.cpp
tcp.cpp
tcppacket.cpp
datacenter/connection_matrix.cpp
datacenter/fat_tree_topology.cpp
datacenter/firstfit.cpp
datacenter/mixnet.cpp
datacenter/fc_topology.cpp
datacenter/flat_topology.cpp
datacenter/os_fattree.cpp
datacenter/agg_os_fattree.cpp
"

SIMAI_FILES="
network_frontend/htsim/AstraSimNetwork.cc
system/AstraParamParse.cc
system/BaseStream.cc
system/BasicEventHandlerData.cc
system/CollectivePhase.cc
system/DMA_Request.cc
system/DataSet.cc
system/IntData.cc
system/LogGP.cc
system/MemBus.cc
system/MemMovRequest.cc
system/MockNcclChannel.cc
system/MockNcclGroup.cc
system/MockNcclLog.cc
system/MyPacket.cc
system/NetworkStat.cc
system/PacketBundle.cc
system/QueueLevelHandler.cc
system/QueueLevels.cc
system/RecvPacketEventHadndlerData.cc
system/RendezvousRecvData.cc
system/RendezvousSendData.cc
system/SendPacketEventHandlerData.cc
system/SharedBusStat.cc
system/SimRecvCaller.cc
system/SimSendCaller.cc
system/StatData.cc
system/StreamBaseline.cc
system/StreamStat.cc
system/Sys.cc
system/Usage.cc
system/UsageTracker.cc
system/calbusbw.cc
system/collective/Algorithm.cc
system/collective/AllToAll.cc
system/collective/DoubleBinaryTreeAllReduce.cc
system/collective/HalvingDoubling.cc
system/collective/NcclTreeFlowModel.cc
system/collective/Ring.cc
system/scheduling/OfflineGreedy.cc
system/topology/BasicLogicalTopology.cc
system/topology/BinaryTree.cc
system/topology/ComplexLogicalTopology.cc
system/topology/DoubleBinaryTreeTopology.cc
system/topology/GeneralComplexTopology.cc
system/topology/LocalRingGlobalBinaryTree.cc
system/topology/LocalRingNodeA2AGlobalDBT.cc
system/topology/LogicalTopology.cc
system/topology/Node.cc
system/topology/RingTopology.cc
system/topology/Torus3D.cc
workload/CSVWriter.cc
workload/Layer.cc
workload/Workload.cc
"

# 构建完整源文件路径列表
SRCS=()
for f in $HTSIM_FILES; do
    SRCS+=("${HTSIM_SRC}/${f}")
done
for f in $SIMAI_FILES; do
    SRCS+=("${SIMAI_SRC}/astra-sim/${f}")
done

# 检查源文件存在
MISSING=0
for src in "${SRCS[@]}"; do
    if [ ! -f "$src" ]; then
        echo "Missing: $src"
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "Error: some source files not found"
    exit 1
fi

echo "Found ${#SRCS[@]} source files, building with ${JOBS} jobs..."

mkdir -p "$BUILD"

# 编译函数
compile_one() {
    local src="$1"
    local rel="${src#${ROOT}/}"
    local obj="${BUILD}/${rel}.o"
    mkdir -p "$(dirname "$obj")"

    # 增量: 源文件比 .o 新才重编
    if [ "$obj" -nt "$src" ] 2>/dev/null; then
        return 0
    fi

    echo "  CC  ${rel}"
    g++ $CXXFLAGS $INCLUDES -c "$src" -o "$obj"
}
export -f compile_one
export ROOT BUILD CXXFLAGS INCLUDES

# 编译
FAIL=0
if [ "$JOBS" -gt 1 ]; then
    printf '%s\n' "${SRCS[@]}" | xargs -P "$JOBS" -I{} bash -c 'compile_one "$@"' _ {} || FAIL=1
else
    for src in "${SRCS[@]}"; do
        compile_one "$src" || FAIL=1
    done
fi

if [ "$FAIL" -eq 1 ]; then
    echo "Error: compilation failed"
    exit 1
fi

# 链接 (用白名单对应的 .o 文件)
echo "Linking..."
OBJ_FILES=()
for src in "${SRCS[@]}"; do
    rel="${src#${ROOT}/}"
    OBJ_FILES+=("${BUILD}/${rel}.o")
done

g++ -std=c++17 -O0 -g "${OBJ_FILES[@]}" -o "$OUT"

echo "Done: $OUT"
