# Notes: This model is run with middle school in mind (in the U.S.)


from dataclasses import dataclass
from dataclasses import replace
from dataclasses import asdict
from collections import defaultdict
import statistics as stats

import numpy as np
import csv
import os

import plot_run_summaries

# for analysis and error checking
# https://chatgpt.com/c/6945a0a5-b57c-8328-aaec-b7c5cfef2879


@dataclass
class Params:
    # Experimental design (drivers)
    n_classes: int = 10          # fixed number of classes
    class_size_cap: int = 30     # treat as ACTUAL class size

    # Derived (auto-set from n_classes and class_size_cap)
    n_students: int = 0
    n_teachers: int = 0

    # Teacher resources
    teacher_time_budget: float = 30.0

    # Learning dynamics
    alpha: float = 0.05  # Learning rate scale
    forgetting: float = 0.002  # daily forgetting proportional to K

    # Run controls
    n_days: int = 90
    seed: int = 4  # run d: 1, run c: 2, run b: 3, run a: 4

    # Teacher skill distribution
    teacher_skill_mean: float = 0.80
    teacher_skill_sd: float = 0.10

    # Disruption model (new in V1)
    inc_c0: float = -2.7  # baseline (lower = fewer incidents): -2.2 for high school
    inc_c_class: float = 0.05  # effect of class size: 0.03 for high school
    inc_c_B: float = 1.0  # effect of student behavior propensity

    base_loss: float = 0.04  # fraction of day lost per incident (0.03 HS)
    max_loss: float = 0.8  # cap on time lost per class per day (0.40 HS)

    # Agency (new in V2)
    agency_amp_min: float = 0.5     # A=0 -> multiplier 0.5
    agency_amp_max: float = 1.0     # A=1 -> multiplier 1.0

    beta_success: float = 0.010     # how much A increases on success
    gamma_failure: float = 0.006    # how much A decreases on failure
    success_eps: float = 1e-6       # what counts as "success" in dK

@dataclass
class StudentAgent:
    sid: int
    K: float  # mastery
    A: float  # agency
    B: float  # behavior propensity


@dataclass
class TeacherAgent:
    tid: int
    skill: float
    time_budget: float



# this is for the daily_all_runs.csv file
OUTPUT_NAMES = [
    "K_mean", "K_p10", "K_p50", "K_p90",
    "A_mean", "A_p10", "A_p50", "A_p90",
    "incidents_total",
    "time_lost_mean",
]

def append_history_csv(filename: str, p: Params, history, run_id: int = 1):
    p_dict = asdict(p)
    param_names = list(p_dict.keys())

    header = ["day", "run_id"] + param_names + OUTPUT_NAMES

    file_exists = False
    try:
        with open(filename, "r", encoding="utf-8") as _:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(filename, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # write header once
        if not file_exists:
            w.writerow(header)

        # append rows
        for row in history:
            day = row[0]
            outputs = list(row[1:])
            params = [p_dict[name] for name in param_names]
            w.writerow([day, run_id] + params + outputs)

# this is for the summary file
OUTPUT_COLS = [
    "K_mean", "K_p10", "K_p50", "K_p90",
    "A_mean", "A_p10", "A_p50", "A_p90",
    "incidents_total",
    "time_lost_mean",
]


def summarize_daily_all_runs(infile: str, outfile: str):
    runs_params = {}                 # run_id -> dict of param columns (strings)
    runs_rows = defaultdict(list)    # run_id -> list of parsed daily rows

    with open(infile, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        if "day" not in header or "run_id" not in header:
            raise ValueError("Expected columns 'day' and 'run_id' in daily_all_runs.csv")

        # Input params are everything except day, run_id, and known outputs
        param_cols = [c for c in header if c not in (["day", "run_id"] + OUTPUT_COLS)]

        for row in reader:
            run_id = int(row["run_id"])
            day = int(row["day"])

            # Store params once per run (from first encountered row)
            if run_id not in runs_params:
                runs_params[run_id] = {c: row[c] for c in param_cols}

            # Parse outputs we’ll summarize
            runs_rows[run_id].append({
                "day": day,
                "K_mean": float(row["K_mean"]),
                "K_p10": float(row["K_p10"]),
                "K_p50": float(row["K_p50"]),
                "K_p90": float(row["K_p90"]),
                "A_mean": float(row["A_mean"]),
                "A_p10": float(row["A_p10"]),
                "A_p50": float(row["A_p50"]),
                "A_p90": float(row["A_p90"]),
                "incidents_total": int(row["incidents_total"]),
                "time_lost_mean": float(row["time_lost_mean"]),
            })

    # Define summary columns (one row per run)
    summary_cols = [
        "n_days",
        "K_mean_start", "A_mean_start",
        "K_mean_end", "K_p10_end", "K_p50_end", "K_p90_end",
        "A_mean_end", "A_p10_end", "A_p50_end", "A_p90_end",
        "K_mean_gain", "A_mean_gain",
        "incidents_avg_per_day", "incidents_total",
        "time_lost_avg", "time_lost_max",
    ]

    out_header = ["run_id"] + param_cols + summary_cols

    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_header)
        writer.writeheader()

        for run_id in sorted(runs_rows.keys()):
            rows = sorted(runs_rows[run_id], key=lambda r: r["day"])
            first = rows[0]
            last = rows[-1]

            n_days = len(rows)
            incidents_total = sum(r["incidents_total"] for r in rows)
            incidents_avg = incidents_total / n_days
            time_lost_avg = sum(r["time_lost_mean"] for r in rows) / n_days
            time_lost_max = max(r["time_lost_mean"] for r in rows)

            summary = {
                "run_id": run_id,
                **runs_params.get(run_id, {}),

                "n_days": n_days,

                "K_mean_start": first["K_mean"],
                "A_mean_start": first["A_mean"],

                "K_mean_end": last["K_mean"],
                "K_p10_end": last["K_p10"],
                "K_p50_end": last["K_p50"],
                "K_p90_end": last["K_p90"],

                "A_mean_end": last["A_mean"],
                "A_p10_end": last["A_p10"],
                "A_p50_end": last["A_p50"],
                "A_p90_end": last["A_p90"],

                "K_mean_gain": last["K_mean"] - first["K_mean"],
                "A_mean_gain": last["A_mean"] - first["A_mean"],

                "incidents_avg_per_day": incidents_avg,
                "incidents_total": incidents_total,

                "time_lost_avg": time_lost_avg,
                "time_lost_max": time_lost_max,
            }

            writer.writerow(summary)

# Converts any real-valued score into a probability between 0 and 1
# (useful for incident likelihood).
# if x is very negative => sigmoid(x) ~ 0
# if x is 0 => sigmoid = .5
# if x is very positive => sigmoid ~ 1
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def init_population(p: Params):
    # Create a repeatable random-number generator;
    # using p.seed means the simulation runs the same way every time for the same settings.
    rng = np.random.default_rng(p.seed)

    # Beta distributions are always between 0 and 1
    # Mean is a / (a + b) or 2 / (2+3) = 2/5 = .4, with a "normalish" distribution
    # variance = (a*b) / ((a+b)^2 * (a + b + 1)) and std = sqrt(variance)
    # Student state variables
    K0 = rng.beta(2.0, 3.0, p.n_students)  # mastery
    B0 = rng.beta(2.0, 3.0, p.n_students)  # behavior propensity
    A0 = rng.beta(2.0, 4.0, p.n_students)  # agency (your updated middle-school choice)

    students = [
        StudentAgent(sid=i, K=float(K0[i]), A=float(A0[i]), B=float(B0[i]))
        for i in range(p.n_students)
    ]

    # Teacher skill distribution (kept as you had it)
    skill = rng.beta(16, 4, p.n_teachers)
    eps = 1e-6
    skill = np.clip(skill, eps, 1 - eps)

    teachers = [
        TeacherAgent(tid=j, skill=float(skill[j]), time_budget=float(p.teacher_time_budget))
        for j in range(p.n_teachers)
    ]

    return rng, students, teachers

def finalize_params(p: Params) -> Params:
    n_students = p.n_classes * p.class_size_cap
    n_teachers = p.n_classes  # teachers = classes
    return replace(p, n_students=n_students, n_teachers=n_teachers)


def make_classes(rng, n_students: int, class_size: int, n_classes: int):
    order = rng.permutation(n_students)

    classes = []
    start = 0
    for _ in range(n_classes):
        classes.append(order[start:start + class_size])
        start += class_size

    return classes

def run(p: Params):
    rng, students, teachers = init_population(p)
    history = []

    for day in range(1, p.n_days + 1):
        incidents_total = 0
        time_lost_total = 0.0

        classes = make_classes(rng, p.n_students, p.class_size_cap, p.n_classes)

        for c_idx, cls in enumerate(classes):
            teacher = teachers[c_idx % p.n_teachers]
            class_size = len(cls)

            attention_per_student = teacher.time_budget / class_size

            # --- incidents (student-level probabilities) ---
            inc_count = 0
            for i in cls:
                s = students[int(i)]
                x = p.inc_c0 + p.inc_c_class * class_size + p.inc_c_B * s.B
                if rng.random() < sigmoid(x):
                    inc_count += 1

            time_loss = min(p.max_loss, p.base_loss * inc_count)
            time_factor = 1.0 - time_loss

            incidents_total += inc_count
            time_lost_total += time_loss * class_size  # student-weighted, same as before

            # --- learning update (agency fixed, but still used as amplifier) ---
            for i in cls:
                s = students[int(i)]

                agency_amp = p.agency_amp_min + (p.agency_amp_max - p.agency_amp_min) * s.A

                dK = (
                    p.alpha
                    * (attention_per_student * time_factor)
                    * teacher.skill
                    * agency_amp
                    * (1.0 - s.K)
                    - p.forgetting * s.K
                )

                s.K = float(np.clip(s.K + dK, 0.0, 1.0))
                # s.A stays fixed (you already removed the update)

        # --- collect stats (same outputs as before) ---
        K_arr = np.array([s.K for s in students], dtype=float)
        A_arr = np.array([s.A for s in students], dtype=float)

        history.append((
            day,
            float(K_arr.mean()),
            float(np.quantile(K_arr, 0.10)),
            float(np.quantile(K_arr, 0.50)),
            float(np.quantile(K_arr, 0.90)),
            float(A_arr.mean()),
            float(np.quantile(A_arr, 0.10)),
            float(np.quantile(A_arr, 0.50)),
            float(np.quantile(A_arr, 0.90)),
            int(incidents_total),
            float(time_lost_total / p.n_students),
        ))

    return history


def aggregate_run_summaries_by_cap(infile: str, outfile: str):
    """
    Collapse run_summaries.csv (one row per run) into scenario_summary.csv
    (one row per class_size_cap) with mean/sd across replications.
    """
    rows = []
    with open(infile, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # group rows by class_size_cap
    groups = defaultdict(list)
    for row in rows:
        cap = int(float(row["class_size_cap"]))
        groups[cap].append(row)

    # metrics that exist in your run_summaries.csv
    metrics = [
        "K_mean_end", "K_p10_end", "K_p50_end", "K_p90_end",
        "A_mean_end", "A_p10_end", "A_p50_end", "A_p90_end",
        "K_mean_gain", "A_mean_gain",
        "incidents_avg_per_day",
        "time_lost_avg", "time_lost_max",
    ]

    out_header = ["class_size_cap", "n_reps"]
    for m in metrics:
        out_header += [f"{m}_mean", f"{m}_sd"]

    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_header)
        w.writeheader()

        for cap in sorted(groups.keys()):
            g = groups[cap]
            out = {"class_size_cap": cap, "n_reps": len(g)}

            for m in metrics:
                vals = [float(r[m]) for r in g if r.get(m, "") != ""]
                out[f"{m}_mean"] = stats.mean(vals) if vals else ""
                out[f"{m}_sd"] = (stats.stdev(vals) if len(vals) >= 2 else 0.0) if vals else ""

            w.writerow(out)


if __name__ == "__main__":
    base = Params()
    out_file = "daily_all_runs.csv"

    # start fresh each time
    for fn in (out_file, "run_summaries.csv", "scenario_summary.csv"):
        if os.path.exists(fn):
            os.remove(fn)

    caps = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    n_reps = 10

    run_id = 0
    for cap in caps:
        for rep in range(n_reps):
            run_id += 1

            # unique seed per (cap, rep) so each replication differs
            seed = base.seed + cap * 1000 + rep

            p0 = replace(base, class_size_cap=cap, seed=seed)
            p = finalize_params(p0)

            hist = run(p)
            append_history_csv(out_file, p, hist, run_id=run_id)

    summarize_daily_all_runs("daily_all_runs.csv", "run_summaries.csv")
    print("Wrote run_summaries.csv")

    aggregate_run_summaries_by_cap("run_summaries.csv", "scenario_summary.csv")
    print("Wrote scenario_summary.csv")

    plot_run_summaries.run_plots()
