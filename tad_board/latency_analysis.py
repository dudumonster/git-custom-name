import csv
import os
import argparse
import matplotlib.pyplot as plt


def read_latency(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return 0.0


def to_int(row, key):
    try:
        return int(float(row[key]))
    except Exception:
        return 0


def plot_per_frame(rows, out_dir, max_frames):
    if not rows:
        return
    rows = rows[:max_frames]
    frame_ids = [to_int(r, "frame_id") for r in rows]
    compute_delay = [to_float(r, "compute_delay") for r in rows]
    encode_delay = [to_float(r, "encode_delay") for r in rows]
    tx_delay = [to_float(r, "tx_delay") for r in rows]
    prop_delay = [to_float(r, "prop_delay") for r in rows]
    total_delay = [to_float(r, "total_delay") for r in rows]

    compute_total = [c + e for c, e in zip(compute_delay, encode_delay)]

    plt.figure(figsize=(12, 6))
    plt.bar(frame_ids, compute_total, label="compute")
    plt.bar(frame_ids, tx_delay, bottom=compute_total, label="tx")
    bottom2 = [c + t for c, t in zip(compute_total, tx_delay)]
    plt.bar(frame_ids, prop_delay, bottom=bottom2, label="prop")
    plt.xlabel("frame_id")
    plt.ylabel("delay (s)")
    plt.title("Latency Breakdown per Frame")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_bar.png"))

    plt.figure(figsize=(12, 6))
    plt.plot(frame_ids, total_delay, label="total")
    plt.plot(frame_ids, compute_total, label="compute")
    plt.plot(frame_ids, tx_delay, label="tx")
    plt.plot(frame_ids, prop_delay, label="prop")
    plt.xlabel("frame_id")
    plt.ylabel("delay (s)")
    plt.title("Latency Trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_line.png"))


def plot_by_quality(rows, out_dir):
    if not rows:
        return
    grouped = {}
    for r in rows:
        q = to_int(r, "jpeg_quality")
        grouped.setdefault(q, []).append(r)

    qualities = sorted(grouped.keys())
    if len(qualities) <= 1:
        return

    mean_compute = []
    mean_tx = []
    mean_prop = []
    mean_total = []
    for q in qualities:
        items = grouped[q]
        compute_delay = [to_float(r, "compute_delay") + to_float(r, "encode_delay") for r in items]
        tx_delay = [to_float(r, "tx_delay") for r in items]
        prop_delay = [to_float(r, "prop_delay") for r in items]
        total_delay = [to_float(r, "total_delay") for r in items]
        mean_compute.append(sum(compute_delay) / max(len(compute_delay), 1))
        mean_tx.append(sum(tx_delay) / max(len(tx_delay), 1))
        mean_prop.append(sum(prop_delay) / max(len(prop_delay), 1))
        mean_total.append(sum(total_delay) / max(len(total_delay), 1))

    plt.figure(figsize=(10, 6))
    plt.plot(qualities, mean_total, marker="o", label="total")
    plt.plot(qualities, mean_compute, marker="o", label="compute")
    plt.plot(qualities, mean_tx, marker="o", label="tx")
    plt.plot(qualities, mean_prop, marker="o", label="prop")
    plt.xlabel("jpeg_quality")
    plt.ylabel("delay (s)")
    plt.title("Latency vs JPEG Quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_quality.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join("latency_logs", "obu_latency.csv"))
    parser.add_argument("--out", default=os.path.join("latency_logs"))
    parser.add_argument("--max-frames", type=int, default=100)
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), csv_path)
    out_dir = args.out
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(__file__), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = read_latency(csv_path)
    if not rows:
        print("No data to plot.")
        return

    plot_per_frame(rows, out_dir, args.max_frames)
    plot_by_quality(rows, out_dir)
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
