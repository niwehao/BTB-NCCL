import csv
import re
from pathlib import Path

TRACE_DIR = Path("/home/weihaoni/BTB-NCCL/Croz/trace")
BANDWIDTH_GB_S = 50.0
BANDWIDTH_B_S = BANDWIDTH_GB_S * 1024**3

ALLREDUCE_TYPES = {"ALLREDUCE"}
RING_ONEWAY_TYPES = {
    "ALLGATHER", "REDUCESCATTER",
    "ALLGATHER_DP_EP", "REDUCESCATTER_DP_EP",
}
A2A_TYPES = {"ALLTOALL"}
P2P_TYPES = {"SENDRECV"}

TP_LAYER_PREFIXES = (
    "attention_column", "attention_row",
    "mlp_column", "mlp_row",
    "embedding_layer", "final_column",
    "cross_entropy",
    "optimizer",
)


def op_time_sec(comm_type, size_bytes):
    if comm_type == "NONE" or size_bytes <= 0:
        return 0.0
    if comm_type in ALLREDUCE_TYPES:
        coef = 2.0
    else:
        coef = 1.0
    return coef * size_bytes / BANDWIDTH_B_S


def classify_group(layer_name, comm_type, tp, pp, ep, dp):
    if comm_type.endswith("_DP_EP"):
        return "DP_EP", max(dp * ep, 1)
    if layer_name.startswith("moe_"):
        return "DP_EP", max(dp * ep, 1)
    if layer_name.startswith("grad_norm"):
        return "DP", max(dp, 1)
    if any(layer_name.startswith(p) for p in TP_LAYER_PREFIXES):
        return "TP", max(tp, 1)
    return "UNKNOWN", max(tp, 1)


def ring_rounds_and_peers(comm_type, group_size):
    if comm_type in ALLREDUCE_TYPES:
        return 2 * max(group_size - 1, 1), 1
    if comm_type in RING_ONEWAY_TYPES:
        return max(group_size - 1, 1), 1
    if comm_type in A2A_TYPES:
        return 1, max(group_size - 1, 1)
    if comm_type in P2P_TYPES:
        return 1, 1
    return max(group_size - 1, 1), 1


def parse_header(line):
    tp = pp = ep = 1
    all_gpus = 0
    m = re.search(r"model_parallel_NPU_group:\s*(\d+)", line)
    if m:
        tp = int(m.group(1))
    m = re.search(r"\bpp:\s*(\d+)", line)
    if m:
        pp = int(m.group(1))
    m = re.search(r"\bep:\s*(\d+)", line)
    if m:
        ep = int(m.group(1))
    m = re.search(r"all_gpus:\s*(\d+)", line)
    if m:
        all_gpus = int(m.group(1))
    dp = max(all_gpus // max(tp * pp * ep, 1), 1)
    return tp, pp, ep, dp, all_gpus


def parse_trace(path):
    with open(path) as f:
        header = f.readline().rstrip("\n")
        tp, pp, ep, dp, all_gpus = parse_header(header)
        _ = f.readline()
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue
            rows.append({
                "layer": parts[0],
                "fwd_type": parts[3], "fwd_size": int(parts[4]),
                "bwd_type": parts[6], "bwd_size": int(parts[7]),
                "dp_type": parts[9], "dp_size": int(parts[10]),
            })
    return header, (tp, pp, ep, dp, all_gpus), rows


def build_timeline(rows, parallel):
    tp, pp, ep, dp, all_gpus = parallel
    events = []
    for r in rows:
        if op_time_sec(r["fwd_type"], r["fwd_size"]) > 0:
            events.append(("fwd", r["layer"], r["fwd_type"], r["fwd_size"]))
    for r in reversed(rows):
        if op_time_sec(r["bwd_type"], r["bwd_size"]) > 0:
            events.append(("bwd", r["layer"], r["bwd_type"], r["bwd_size"]))
    for r in rows:
        if op_time_sec(r["dp_type"], r["dp_size"]) > 0:
            events.append(("dp", r["layer"], r["dp_type"], r["dp_size"]))
    enriched = []
    for phase, layer, ct, sz in events:
        group_name, group_size = classify_group(layer, ct, tp, pp, ep, dp)
        rounds, peers = ring_rounds_and_peers(ct, group_size)
        chunk = sz // max(group_size, 1)
        t = op_time_sec(ct, sz)
        enriched.append({
            "phase": phase, "layer": layer, "comm_type": ct, "comm_size": sz,
            "time_sec": t,
            "group_name": group_name, "group_size": group_size,
            "concurrent_send_peers": peers, "per_step_chunk_bytes": chunk,
            "total_rounds": rounds,
        })
    return enriched


def main():
    for txt in sorted(TRACE_DIR.glob("*.txt")):
        header, parallel, rows = parse_trace(txt)
        tp, pp, ep, dp, all_gpus = parallel
        events = build_timeline(rows, parallel)
        total_time = sum(e["time_sec"] for e in events)
        if total_time == 0:
            print(f"{txt.name}: no comm events, skipped")
            continue

        out_path = txt.with_name(txt.stem + "_timeline.csv")
        cum = 0.0
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "step_idx", "phase", "layer_name", "comm_type", "comm_size",
                "time_sec", "cum_pct",
                "tp", "pp", "ep", "dp",
                "group_name", "group_size",
                "concurrent_send_peers", "per_step_chunk_bytes", "total_rounds",
            ])
            for i, e in enumerate(events):
                cum += e["time_sec"]
                pct = cum / total_time * 100.0
                w.writerow([
                    i, e["phase"], e["layer"], e["comm_type"], e["comm_size"],
                    f"{e['time_sec']:.6f}", f"{pct:.4f}",
                    tp, pp, ep, dp,
                    e["group_name"], e["group_size"],
                    e["concurrent_send_peers"], e["per_step_chunk_bytes"], e["total_rounds"],
                ])

        print(f"{txt.name}: tp={tp} pp={pp} ep={ep} dp={dp} all_gpus={all_gpus} "
              f"events={len(events)} total_time={total_time:.3f}s -> {out_path.name}")


if __name__ == "__main__":
    main()
