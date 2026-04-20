import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/home/weihaoni/BTB-NCCL/Croz/alibaba-lingjun-dataset-2023/data")
JOB_CSV = DATA_DIR / "job.csv"
WORKER_CSV = DATA_DIR / "worker.csv"
OUT_CSV = Path("/home/weihaoni/BTB-NCCL/Croz/jobs_real_models.csv")

EXCLUDED_MODELS = {
    "unknown", "rl", "preprocess", "sidecar", "concat",
    "reward", "image", "layer_attn", "multigen", "dota",
    "cognitive",
}


def aggregate_gpu_per_job(worker_df):
    gpu_map = {}
    for job_name, grp in worker_df.groupby("job_name"):
        total = 0
        for res_str in grp["RES"].dropna():
            try:
                res = json.loads(res_str)
            except Exception:
                continue
            total += int(res.get("nvidia.com/gpu", 0) or 0)
        gpu_map[job_name] = total
    return gpu_map


def main():
    df = pd.read_csv(JOB_CSV, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    before = len(df)
    kept = df[~df["model"].isin(EXCLUDED_MODELS) & df["model"].notna()].copy()
    print(f"Rows kept after model filter: {len(kept)} / {before}")

    worker_df = pd.read_csv(WORKER_CSV, low_memory=False)
    worker_df = worker_df.loc[:, ~worker_df.columns.str.startswith("Unnamed")]
    gpu_map = aggregate_gpu_per_job(worker_df)

    kept["total_gpu"] = kept["job_name"].map(gpu_map).fillna(0).astype(int)

    out = kept[["model", "total_gpu"]]
    out.to_csv(OUT_CSV, index=False)
    print(f"Written {len(out)} rows to {OUT_CSV}")
    print("\nSample (head 10):")
    print(out.head(10).to_string(index=False))
    print("\nGPU total stats per model:")
    print(out.groupby("model")["total_gpu"].describe()[["count", "min", "mean", "max"]].round(1))


if __name__ == "__main__":
    main()
