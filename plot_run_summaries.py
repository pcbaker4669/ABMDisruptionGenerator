# plot_run_summaries.py (1D sweep: class_size_cap only)
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _to_int(x):
    try:
        return int(float(x))
    except Exception:
        return 0


def journal_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Helpful for journal PDFs (avoid Type 3 fonts in many setups)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_both(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def load_rows(csv_path: str):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def run_plots():
    journal_style()

    in_csv = "run_summaries.csv"
    out_dir = Path("figures_1d_class_size")

    rows = load_rows(in_csv)

    # Parse only what we need for a 1D sweep
    parsed = []
    for r in rows:
        parsed.append({
            "class_size_cap": _to_int(r["class_size_cap"]),
            "K_mean_end": _to_float(r["K_mean_end"]),
            "K_p10_end": _to_float(r["K_p10_end"]),
            "K_p50_end": _to_float(r["K_p50_end"]),
            "K_p90_end": _to_float(r["K_p90_end"]),
            "A_mean_end": _to_float(r["A_mean_end"]),
            "A_p10_end": _to_float(r["A_p10_end"]),
            "A_p50_end": _to_float(r["A_p50_end"]),
            "A_p90_end": _to_float(r["A_p90_end"]),
            "incidents_avg_per_day": _to_float(r["incidents_avg_per_day"]),
            "time_lost_avg": _to_float(r["time_lost_avg"]),
            "time_lost_max": _to_float(r["time_lost_max"]),
        })

    parsed.sort(key=lambda d: d["class_size_cap"])
    x = [d["class_size_cap"] for d in parsed]

    # --- Figure 1: End-of-run mastery distribution ---
    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.gca()
    ax.plot(x, [d["K_mean_end"] for d in parsed], marker="o", label="K_mean_end")
    ax.plot(x, [d["K_p10_end"] for d in parsed], marker="o", label="K_p10_end")
    ax.plot(x, [d["K_p50_end"] for d in parsed], marker="o", label="K_p50_end")
    ax.plot(x, [d["K_p90_end"] for d in parsed], marker="o", label="K_p90_end")
    ax.set_xlabel("Class size cap (students)")
    ax.set_ylabel("End-of-run mastery (K)")
    ax.set_title("End-of-run mastery vs class size cap")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_both(fig, out_dir, "fig1_mastery_end_vs_class_size")

    # --- Figure 2: End-of-run agency distribution ---
    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.gca()
    ax.plot(x, [d["A_mean_end"] for d in parsed], marker="o", label="A_mean_end")
    ax.plot(x, [d["A_p10_end"] for d in parsed], marker="o", label="A_p10_end")
    ax.plot(x, [d["A_p50_end"] for d in parsed], marker="o", label="A_p50_end")
    ax.plot(x, [d["A_p90_end"] for d in parsed], marker="o", label="A_p90_end")
    ax.set_xlabel("Class size cap (students)")
    ax.set_ylabel("End-of-run agency (A)")
    ax.set_title("End-of-run agency vs class size cap")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_both(fig, out_dir, "fig2_agency_end_vs_class_size")

    # --- Figure 3: Incidents avg/day vs class size cap (own figure) ---
    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.gca()
    ax.plot(x, [d["incidents_avg_per_day"] for d in parsed], marker="o")
    ax.set_xlabel("Class size cap (students)")
    ax.set_ylabel("Incidents avg/day")
    ax.set_title("Incidents vs class size cap")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    save_both(fig, out_dir, "fig3_incidents_vs_class_size")

    # --- Figure 4: time_lost_avg vs class size cap (own figure; key metric) ---
    y_loss = [d["time_lost_avg"] for d in parsed]
    y_max = [d["time_lost_max"] for d in parsed]

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.gca()
    ax.plot(x, y_loss, marker="o", label="time_lost_avg")

    # Optional but useful context: shows whether you're approaching the cap
    ax.plot(x, y_max, marker="o", linestyle="--", label="time_lost_max")

    ax.set_xlabel("Class size cap (students)")
    ax.set_ylabel("Avg instructional time lost (fraction of day)")
    ax.set_title("Instructional time loss vs class size cap")
    ax.set_xticks(x)
    ax.set_ylim(0.0, max(y_max + y_loss) * 1.1 if (y_max and y_loss) else 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_both(fig, out_dir, "fig4_time_lost_vs_class_size")

    # --- Figure 5: Marginal effect on mastery (Δ between adjacent caps) ---
    dK = []
    x_mid = []
    for i in range(1, len(parsed)):
        dK.append(parsed[i]["K_mean_end"] - parsed[i - 1]["K_mean_end"])
        x_mid.append(parsed[i]["class_size_cap"])

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.gca()
    ax.plot(x_mid, dK, marker="o")
    ax.set_xlabel("Class size cap (students)")
    ax.set_ylabel("Δ K_mean_end (adjacent step)")
    ax.set_title("Marginal change in end mastery across class-size steps")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    save_both(fig, out_dir, "fig5_marginal_dK_end_vs_class_size")

    print(f"Done. Wrote PDF+PNG figures to: {out_dir.resolve()}")