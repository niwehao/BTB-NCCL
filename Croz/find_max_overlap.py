import json
import random
import re
import pandas as pd
from pathlib import Path

random.seed(42)

DATA_DIR = Path("/home/weihaoni/BTB-NCCL/Croz/alibaba-lingjun-dataset-2023/data")
TRACE_DIR = Path("/home/weihaoni/BTB-NCCL/Croz/trace")
JOB_CSV = DATA_DIR / "job.csv"
WORKER_CSV = DATA_DIR / "worker.csv"
OUT_CSV = Path("/home/weihaoni/BTB-NCCL/Croz/max_overlap_jobs.csv")
OUT_CSV_ALL = Path("/home/weihaoni/BTB-NCCL/Croz/max_overlap_jobs_all_matched.csv")


def parse_memory_to_gib(mem_str):
    if mem_str is None:
        return 0.0
    m = re.match(r"^\s*([\d.]+)\s*([A-Za-z]*)\s*$", str(mem_str))
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = m.group(2).lower()
    scale = {
        "": 1 / (1024 ** 3), "b": 1 / (1024 ** 3),
        "ki": 1 / (1024 ** 2), "kib": 1 / (1024 ** 2), "k": 1 / (1024 ** 2),
        "mi": 1 / 1024, "mib": 1 / 1024, "m": 1 / 1024,
        "gi": 1.0, "gib": 1.0, "g": 1.0,
        "ti": 1024.0, "tib": 1024.0, "t": 1024.0,
    }.get(unit, 1.0)
    return val * scale


def aggregate_worker_res(worker_df):
    agg = {}
    for job_name, grp in worker_df.groupby("job_name"):
        worker_count = len(grp)
        total_gpu = 0
        total_cpu = 0.0
        total_mem_gib = 0.0
        has_rdma = False
        for res_str in grp["RES"].dropna():
            try:
                res = json.loads(res_str)
            except Exception:
                continue
            total_gpu += int(res.get("nvidia.com/gpu", 0) or 0)
            cpu_str = res.get("cpu", "0")
            try:
                total_cpu += float(cpu_str) if "m" not in str(cpu_str) else float(str(cpu_str).rstrip("m")) / 1000.0
            except Exception:
                pass
            total_mem_gib += parse_memory_to_gib(res.get("memory"))
            if int(res.get("koordinator.sh/rdma", 0) or 0) > 0:
                has_rdma = True
        agg[job_name] = {
            "worker_count": worker_count,
            "total_gpu": total_gpu,
            "total_cpu": total_cpu,
            "total_memory_GiB": round(total_mem_gib, 2),
            "has_rdma": has_rdma,
        }
    return agg


def assign_state(total_gpu):
    if total_gpu < 10:
        p_comm = 0.15
    elif total_gpu < 1000:
        p_comm = 0.30
    else:
        p_comm = 0.50
    return "Communication" if random.random() < p_comm else "Computation"


def load_timelines():
    timelines = {}
    for csv_path in sorted(TRACE_DIR.glob("*_timeline.csv")):
        m = re.search(r"(\d+)gpu", csv_path.stem)
        if not m:
            continue
        gpu = int(m.group(1))
        tl_df = pd.read_csv(csv_path)
        src_txt = csv_path.name.replace("_timeline.csv", ".txt")
        timelines[gpu] = {
            "df": tl_df,
            "cum_pct": tl_df["cum_pct"].to_numpy(),
            "source_txt": src_txt,
        }
    return timelines


def match_event(timelines, total_gpu, target_pct):
    if not timelines:
        return None
    gpu_keys = sorted(timelines.keys())
    best_gpu = min(gpu_keys, key=lambda g: abs(g - total_gpu))
    tl = timelines[best_gpu]
    cum = tl["cum_pct"]
    idx = int((cum - target_pct).__abs__().argmin())
    row = tl["df"].iloc[idx]
    return {
        "matched_trace": tl["source_txt"],
        "matched_phase": row["phase"],
        "matched_layer": row["layer_name"],
        "comm_type": row["comm_type"],
        "comm_size": int(row["comm_size"]),
        "tp": int(row["tp"]),
        "pp": int(row["pp"]),
        "ep": int(row["ep"]),
        "dp": int(row["dp"]),
        "group_name": row["group_name"],
        "group_size": int(row["group_size"]),
        "concurrent_send_peers": int(row["concurrent_send_peers"]),
        "per_step_chunk_bytes": int(row["per_step_chunk_bytes"]),
        "total_rounds": int(row["total_rounds"]),
    }


def main():
    job_df = pd.read_csv(JOB_CSV, low_memory=False)
    job_df = job_df.loc[:, ~job_df.columns.str.startswith("Unnamed")]

    job_df["start"] = pd.to_datetime(job_df["gmt_job_running"], errors="coerce")
    job_df["end"] = pd.to_datetime(job_df["gmt_job_finished"], errors="coerce")
    ran = job_df.dropna(subset=["start", "end"]).copy()
    ran = ran[ran["end"] > ran["start"]]
    print(f"Jobs with valid running+finished times: {len(ran)} / {len(job_df)}")

    worker_df_all = pd.read_csv(WORKER_CSV, low_memory=False)
    worker_df_all = worker_df_all.loc[:, ~worker_df_all.columns.str.startswith("Unnamed")]
    worker_agg_all = aggregate_worker_res(worker_df_all)
    before = len(ran)
    ran = ran[ran["job_name"].apply(lambda n: worker_agg_all.get(n, {}).get("total_gpu", 0) >= 8)]
    print(f"After filtering total_gpu<8: {len(ran)} / {before}")

    events = []
    for _, row in ran.iterrows():
        events.append((row["start"], 1, row["id"]))
        events.append((row["end"], -1, row["id"]))
    events.sort(key=lambda x: (x[0], -x[1]))

    active = set()
    peak = 0
    peak_time = None
    peak_active = set()
    for t, delta, jid in events:
        if delta == 1:
            active.add(jid)
            if len(active) > peak:
                peak = len(active)
                peak_time = t
                peak_active = set(active)
        else:
            active.discard(jid)

    print(f"Peak concurrent running jobs: {peak} at {peak_time}")

    picked = ran[ran["id"].isin(peak_active)].copy()
    picked = picked[picked["start"] <= peak_time]
    picked = picked[picked["end"] > peak_time]
    print(f"Selected {len(picked)} jobs overlapping at peak moment")

    timelines = load_timelines()
    print(f"Loaded {len(timelines)} timelines: {sorted(timelines.keys())} GPUs")

    rows = []
    for _, r in picked.iterrows():
        a = worker_agg_all.get(r["job_name"], {})
        total = (r["end"] - r["start"]).total_seconds()
        elapsed = (peak_time - r["start"]).total_seconds()
        progress_pct = round(elapsed / total * 100.0, 2) if total > 0 else 0.0
        total_gpu = a.get("total_gpu", 0)
        state = assign_state(total_gpu)
        match = match_event(timelines, total_gpu, progress_pct) if state == "Communication" else None
        rows.append({
            "job_id": r["id"],
            "job_name": r["job_name"],
            "model": r["model"],
            "gmt_job_running": r["gmt_job_running"],
            "gmt_job_finished": r["gmt_job_finished"],
            "progress_pct_at_peak": progress_pct,
            "worker_count": a.get("worker_count", 0),
            "total_gpu": total_gpu,
            "total_cpu": a.get("total_cpu", 0.0),
            "total_memory_GiB": a.get("total_memory_GiB", 0.0),
            "has_rdma": a.get("has_rdma", False),
            "state": state,
            "matched_trace": match["matched_trace"] if match else "",
            "matched_phase": match["matched_phase"] if match else "",
            "matched_layer": match["matched_layer"] if match else "",
            "comm_type": match["comm_type"] if match else "",
            "comm_size": match["comm_size"] if match else "",
            "tp": match["tp"] if match else "",
            "pp": match["pp"] if match else "",
            "ep": match["ep"] if match else "",
            "dp": match["dp"] if match else "",
            "group_name": match["group_name"] if match else "",
            "group_size": match["group_size"] if match else "",
            "concurrent_send_peers": match["concurrent_send_peers"] if match else "",
            "per_step_chunk_bytes": match["per_step_chunk_bytes"] if match else "",
            "total_rounds": match["total_rounds"] if match else "",
        })

    out_df = pd.DataFrame(rows).sort_values(["total_gpu", "job_id"], ascending=[False, True])
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Written {len(out_df)} rows to {OUT_CSV}")
    print(out_df.head(10).to_string(index=False))

    all_rows = []
    for _, r in picked.iterrows():
        a = worker_agg_all.get(r["job_name"], {})
        total = (r["end"] - r["start"]).total_seconds()
        elapsed = (peak_time - r["start"]).total_seconds()
        progress_pct = round(elapsed / total * 100.0, 2) if total > 0 else 0.0
        total_gpu = a.get("total_gpu", 0)
        match = match_event(timelines, total_gpu, progress_pct)
        all_rows.append({
            "job_id": r["id"],
            "job_name": r["job_name"],
            "model": r["model"],
            "gmt_job_running": r["gmt_job_running"],
            "gmt_job_finished": r["gmt_job_finished"],
            "progress_pct_at_peak": progress_pct,
            "worker_count": a.get("worker_count", 0),
            "total_gpu": total_gpu,
            "total_cpu": a.get("total_cpu", 0.0),
            "total_memory_GiB": a.get("total_memory_GiB", 0.0),
            "has_rdma": a.get("has_rdma", False),
            "matched_trace": match["matched_trace"] if match else "",
            "matched_phase": match["matched_phase"] if match else "",
            "matched_layer": match["matched_layer"] if match else "",
            "comm_type": match["comm_type"] if match else "",
            "comm_size": match["comm_size"] if match else "",
            "tp": match["tp"] if match else "",
            "pp": match["pp"] if match else "",
            "ep": match["ep"] if match else "",
            "dp": match["dp"] if match else "",
            "group_name": match["group_name"] if match else "",
            "group_size": match["group_size"] if match else "",
            "concurrent_send_peers": match["concurrent_send_peers"] if match else "",
            "per_step_chunk_bytes": match["per_step_chunk_bytes"] if match else "",
            "total_rounds": match["total_rounds"] if match else "",
        })
    all_df = pd.DataFrame(all_rows).sort_values(["total_gpu", "job_id"], ascending=[False, True])
    all_df.to_csv(OUT_CSV_ALL, index=False)
    print(f"\nWritten {len(all_df)} rows to {OUT_CSV_ALL} (all jobs matched regardless of state)")


if __name__ == "__main__":
    main()
