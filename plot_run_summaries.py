# plot_run_summaries.py (1D sweep: class_size_cap only; scenario_summary.csv with replications)
import csv
import math
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


def _get_mean_sd(row: dict, base_key: str):
    """
    Supports BOTH formats:
      - scenario_summary.csv: <key>_mean, <key>_sd, plus n_reps
      - run_summaries.csv:   <key> (no sd)
    Returns: (mean, sd, n_reps)
    """
    n_reps = _to_int(row.get("n_reps", "1"))
    if f"{base_key}_mean" in row:
        mean = _to_float(row.get(f"{base_key}_mean", "nan"))
        sd = _to_float(row.get(f"{base_key}_sd", "nan"))
        return mean, sd, n_reps

    # fallback (single-run summaries)
    mean = _to_float(row.get(base_key, "nan"))
    return mean, 0.0, 1


def _yerr_from_sd(sd_list, n_list, mode="ci95"):
    """
    mode:
      - "ci95": 1.96 * (sd/sqrt(n))
      - "sd":   sd
      - "none": no error bars
    """
    yerr = []
    for sd, n in zip(sd_list, n_list):
        n = max(1, int(n))
        if mode == "none":
            yerr.append(0.0)
        elif mode == "sd":
            yerr.append(sd)
        else:
            se = sd / math.sqrt(n)
            yerr.append(1.96 * se)
    return yerr


def run_plots():
    journal_style()

    in_csv = "scenario_summary.csv"
    out_dir = Path("figures_1d_class_size")

    rows = load_rows(in_csv)
    if not rows:
        print(f"No rows found in {in_csv}")
        return

    # Parse what we need for a 1D sweep (means + SDs + n_reps)
    parsed = []
    for r in rows:
        cap = _to_int(r.get("class_size_cap", "0"))

        K_mean, K_mean_sd, nrep = _get_mean_sd(r, "K_mean_end")
        K_p10,  K_p10_sd, _    = _get_mean_sd(r, "K_p10_end")
        K_p50,  K_p50_sd, _    = _get_mean_sd(r, "K_p50_end")
        K_p90,  K_p90_sd, _    = _get_mean_sd(r, "K_p90_end")

        A_mean, A_mean_sd, _   = _get_mean_sd(r, "A_mean_end")
        A_p10,  A_p10_sd, _    = _get_mean_sd(r, "A_p10_end")
        A_p50,  A_p50_sd, _    = _get_mean_sd(r, "A_p50_end")
        A_p90,  A_p90_sd, _    = _get_mean_sd(r, "A_p90_end")

        inc, inc_sd, _         = _get_mean_sd(r, "incidents_avg_per_day")
        tavg, tavg_sd, _       = _get_mean_sd(r, "time_lost_avg")
        tmax, tmax_sd, _       = _get_mean_sd(r, "time_lost_max")

        parsed.append({
            "class_size_cap": cap,
            "n_reps": nrep,

            "K_mean_end": K_mean, "K_mean_end_sd": K_mean_sd,
            "K_p10_end": K_p10,   "K_p10_end_sd": K_p10_sd,
            "K_p50_end": K_p50,   "K_p50_end_sd": K_p50_sd,
            "K_p90_end": K_p90,   "K_p90_end_sd": K_p90_sd,

            "A_mean_end": A_mean, "A_mean_end_sd": A_mean_sd,
            "A_p10_end": A_p10,   "A_p10_end_sd": A_p10_sd,
            "A_p50_end": A_p50,   "A_p50_end_sd": A_p50_sd,
            "A_p90_end": A_p90,   "A_p90_end_sd": A_p90_sd,

            "incidents_avg_per_day": inc, "incidents_avg_per_day_sd": inc_sd,
            "time_lost_avg": tavg,        "time_lost_avg_sd": tavg_sd,
            "time_lost_max": tmax,        "time_lost_max_sd": tmax_sd,
        })

    parsed.sort(key=lambda d: d["class_size_cap"])
    x = [d["class_size_cap"] for d in parsed]
    nreps = [d["n_reps"] for d in parsed]

    # Choose error bar style (ci95 looks best for “replications per scenario”)
    err_mode = "ci95"

    # --- Figure 1: End-of-run mastery (mean + band p10..p90) ---
    K_mean = [d["K_mean_end"] for d in parsed]
    K_mean_sd = [d["K_mean_end_sd"] for d in parsed]
    K_mean_yerr = _yerr_from_sd(K_mean_sd, nreps, mode=err_mode)

    K_p10 = [d["K_p10_end"] for d in parsed]
    K_p50 = [d["K_p50_end"] for d in parsed]
    K_p90 = [d["K_p90_end"] for d in parsed]

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.gca()

    ax.fill_between(x, K_p10, K_p90, alpha=0.15, label="K band (p10..p90)")
    ax.errorbar(x, K_mean, yerr=K_mean_yerr, fmt="o-", capsize=3, label=f"K_mean_end ({err_mode})")
    ax.plot(x, K_p50, marker="o", linestyle="--", label="K_p50_end")

    ax.set_xlabel("Class size cap (students)")
    ax.set_ylabel("End-of-run mastery (K)")
    ax.set_title("End-of-run mastery vs class size cap")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_both(fig, out_dir, "fig1_mastery_end_vs_class_size")

    # --- Figure 2: Incidents avg/day vs class size cap (own figure) ---
    inc = [d["incidents_avg_per_day"] for d in parsed]
    inc_sd = [d["incidents_avg_per_day_sd"] for d in parsed]
    inc_yerr = _yerr_from_sd(inc_sd, nreps, mode=err_mode)

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.gca()
    ax.errorbar(x, inc, yerr=inc_yerr, fmt="o-", capsize=3, label=f"Incidents avg/day ({err_mode})")
    ax.set_xlabel("Class size cap (students)")
    ax.set_ylabel("Incidents avg/day")
    ax.set_title("Incidents vs class size")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_both(fig, out_dir, "fig2_incidents_vs_class_size")

    # --- Figure 3: time_lost_avg vs class size cap (own figure; key metric) ---
    tavg = [d["time_lost_avg"] for d in parsed]
    tavg_sd = [d["time_lost_avg_sd"] for d in parsed]
    tavg_yerr = _yerr_from_sd(tavg_sd, nreps, mode=err_mode)

    tmax = [d["time_lost_max"] for d in parsed]

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.gca()

    ax.errorbar(x, tavg, yerr=tavg_yerr, fmt="o-", capsize=3, label=f"time_lost_avg ({err_mode})")
    ax.plot(x, tmax, marker="o", linestyle="--", label="time_lost_max")

    ax.set_xlabel("Class size cap (students)")
    ax.set_ylabel("Instructional time lost (fraction of day)")
    ax.set_title("Instructional time loss vs class size cap")
    ax.set_xticks(x)
    ax.set_ylim(0.0, max(tmax + tavg) * 1.1 if (tmax and tavg) else 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_both(fig, out_dir, "fig3_time_lost_vs_class_size")

    # --- Figure 4: Marginal effect on mastery (Δ between adjacent caps), with CI if available ---
    # Marginal drop in mastery across adjacent class-size scenarios.
    # Helps reveal nonlinear "tipping" regions where added students amplify disruption/time-loss
    # enough that the per-step mastery penalty grows.

    dK = []
    dK_yerr = []
    x_mid = []

    # error propagation for differences: se(difference)=sqrt(se1^2 + se0^2)
    # using SE from K_mean_end_sd / sqrt(n_reps), then ci95 multiplier
    for i in range(1, len(parsed)):
        x_mid.append(parsed[i]["class_size_cap"])
        dK.append(parsed[i]["K_mean_end"] - parsed[i - 1]["K_mean_end"])

        sd1 = parsed[i]["K_mean_end_sd"]
        sd0 = parsed[i - 1]["K_mean_end_sd"]
        n1 = max(1, parsed[i]["n_reps"])
        n0 = max(1, parsed[i - 1]["n_reps"])
        se = math.sqrt((sd1 / math.sqrt(n1)) ** 2 + (sd0 / math.sqrt(n0)) ** 2)
        dK_yerr.append(1.96 * se if err_mode == "ci95" else (math.sqrt(sd1**2 + sd0**2) if err_mode == "sd" else 0.0))

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.gca()
    ax.errorbar(x_mid, dK, yerr=dK_yerr, fmt="o-", capsize=3)
    ax.set_xlabel("Class size cap (students)")
    ax.set_ylabel("Δ K_mean_end (adjacent step)")
    ax.set_title("Marginal change in end mastery across class-size steps")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    save_both(fig, out_dir, "fig4_marginal_dK_end_vs_class_size")

    print(f"Done. Wrote PDF+PNG figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    run_plots()
