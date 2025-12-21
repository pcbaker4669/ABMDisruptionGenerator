# todo: This model is run with middle school in mind (in the U.S.)
#  There should be an association between agency and environment (incidents), I've
#  seen where students with good agency lose agency in a chaotic environment
#  I believe students with high agency will not be apt to lose agency
#  I believe students that don't do well will lose agency
#  Currently, agency nudges up on successes and down on failures, probably too simplistic

from dataclasses import dataclass
from dataclasses import replace
from dataclasses import asdict
from collections import defaultdict

import numpy as np
import csv
import os

import plot_run_summaries
import plot_run_summaries as my_plots

# for analysis and error checking
# https://chatgpt.com/c/6945a0a5-b57c-8328-aaec-b7c5cfef2879


@dataclass
class Params:
    # population
    n_students: int = 120
    n_teachers: int = 6

    # policy lever
    class_size_cap: int = 30

    # Teacher resources
    teacher_time_budget: float = 30.0 # attention units per teacher per day

    # Learning dynamics
    alpha: float = 0.05  # Learning rate scale
    forgetting: float = 0.002  # daily forgetting proportional to K

    # Run controls
    n_days: int = 90
    seed: int = 1

    # Teacher skill distribution
    teacher_skill_mean: float = 0.80
    teacher_skill_sd: float = 0.10

    # Disruption model (new in V1)
    inc_c0: float = -1.7  # baseline (lower = fewer incidents): -2.2 for high school
    inc_c_class: float = 0.05  # effect of class size: 0.03 for high school
    inc_c_B: float = 1.0  # effect of student behavior propensity (todo: forgot)

    base_loss: float = 0.04  # fraction of day lost per incident (0.03 HS)
    max_loss: float = 0.4  # cap on time lost per class per day (0.40 HS)

    # Agency (new in V2)
    agency_amp_min: float = 0.5     # A=0 -> multiplier 0.5
    agency_amp_max: float = 1.0     # A=1 -> multiplier 1.0

    beta_success: float = 0.010     # how much A increases on success
    gamma_failure: float = 0.006    # how much A decreases on failure
    success_eps: float = 1e-6       # what counts as "success" in dK

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

    # Student mastery starts low-to-mid, beta distributions are always between 0 and 1
    # Mean is a / (a + b) or 2 / (2+3) = 2/5 = .4, with a "normalish" distribution
    # variance = (a*b) / ((a+b)^2 * (a + b + 1)) and std = sqrt(variance)
    K = rng.beta(2.0, 3.0, p.n_students)

    # Draw each teacher’s skill from a normal-ish distribution with mean .8
    # "skill" has an array or teacher skills (0, 1]
    skill = rng.beta(16, 4, p.n_teachers)  # mean = 16/(20)=0.80, fairly tight
    # tiny clip to ensure no teacher is exactly zero
    eps = 1e-6
    skill = np.clip(skill, eps, 1-eps)

    # mean 2 / 5 = .4
    B = rng.beta(2.0, 3.0, p.n_students)  # Student behavior propensity in [0,1]

    # mean 2 / 4.5
    A = rng.beta(2.0, 2.5, p.n_students)  # Student agency in [0,1]

    return rng, K, A, B, skill

def make_classes(rng, n_students: int, class_size_cap: int):
    """
      Balanced class assignment (removes tiny remainder classes).

      - Uses the *minimum* number of classes needed given the cap:
          n_classes = ceil(n_students / class_size_cap)
      - Then spreads students as evenly as possible across those classes,
        so class sizes differ by at most 1 student.

      This prevents artifacts where one cap creates a very small class
      (e.g., cap=28 -> 28,28,28,28,8) that boosts outcomes unrealistically.
      """
    # random shuffle of the student indices.
    # If n_students = 5, something like: array([3, 0, 4, 1, 2])
    order = rng.permutation(n_students)

    # Minimum number of classes needed so nobody exceeds the cap
    n_classes = (n_students + class_size_cap - 1) // class_size_cap  # ceil division

    # Evenly distribute students across classes
    base_size = n_students // n_classes
    remainder = n_students % n_classes  # first 'remainder' classes get +1 student

    # "classes" becomes a list of arrays; each array holds the student IDs assigned
    # to one class for today (up to class_size_cap students per class).
    classes = []
    start = 0
    for i in range(n_classes):
        size = base_size + (1 if i < remainder else 0)
        # Safety check: should always hold with the chosen n_classes
        if size > class_size_cap:
            raise ValueError(f"Balanced class size {size} exceeds cap {class_size_cap}")
        classes.append(order[start:start + size])
        start += size

    return classes

def run(p: Params):
    rng, K, A, B, skill = init_population(p)

    history = []  # (day, K_mean, K_p10, K_p50, K_p90)

    for day in range(1, p.n_days + 1):
        incidents_total = 0
        time_lost_total = 0.0
        classes = make_classes(rng, p.n_students, p.class_size_cap)

        for c_idx, cls in enumerate(classes):
            t = c_idx % p.n_teachers
            class_size = len(cls)

            attention_per_student = p.teacher_time_budget / class_size

            # incidents create time loss for the whole class
            # Probability a student causes an incident today
            x = p.inc_c0 + p.inc_c_class * class_size + p.inc_c_B * B[cls]
            p_inc = sigmoid(x)

            incidents = rng.random(class_size) < p_inc
            inc_count = int(incidents.sum())

            time_loss = min(p.max_loss, p.base_loss * inc_count)
            time_factor = 1.0 - time_loss  # remaining usable instruction time

            incidents_total += inc_count
            time_lost_total += time_loss * class_size  # student-weighted

            # Agency multiplier: maps A in [0,1] to [agency_amp_min, agency_amp_max]
            agency_amp = p.agency_amp_min + (p.agency_amp_max - p.agency_amp_min) * A[cls]

            # Learning is reduced when time is lost to disruption
            dK = (p.alpha * (attention_per_student * time_factor) * skill[t] * agency_amp *
                  (1.0 - K[cls]) - p.forgetting * K[cls])

            # Agency update: success nudges A up, failure nudges A down
            success = (dK > p.success_eps)
            dA = np.where(success, p.beta_success, -p.gamma_failure)

            K[cls] = np.clip(K[cls] + dK, 0.0, 1.0)
            A[cls] = np.clip(A[cls] + dA, 0.0, 1.0)

        history.append((
            day,  # Day number in the simulation
            float(K.mean()),  # Average mastery across all students
            float(np.quantile(K, 0.10)),  # 10th percentile mastery (lower-performing tail)
            float(np.quantile(K, 0.50)),  # Median mastery
            float(np.quantile(K, 0.90)),  # 90th percentile mastery (upper-performing tail)
            float(A.mean()),  # Average agency across all students
            float(np.quantile(A, 0.10)),  # 10th percentile agency (lowest-agency tail)
            float(np.quantile(A, 0.50)),  # Median agency
            float(np.quantile(A, 0.90)),  # 90th percentile agency (highest-agency tail)
            int(incidents_total),  # Total disruptive incidents across all classes today
            float(time_lost_total / p.n_students),  # Average instructional time lost per student today
        ))

    return history


if __name__ == "__main__":
    base = Params()
    out_file = "daily_all_runs.csv"
    # optional: start fresh each time by deleting the old file
    # (comment out if you want to keep appending)
    if os.path.exists(out_file):
        os.remove(out_file)

        # Define scenarios (10 runs). Change these however you like.
    scenarios = [
        {"class_size_cap": 12 },
        {"class_size_cap": 14 },
        {"class_size_cap": 16 },
        {"class_size_cap": 18 },
        {"class_size_cap": 20 },
        {"class_size_cap": 22 },
        {"class_size_cap": 24 },
        {"class_size_cap": 26 },
        {"class_size_cap": 28 },
        {"class_size_cap": 30 },
    ]

    for run_id, overrides in enumerate(scenarios, start=1):
        p = replace(base, **overrides)  # <-- override ANY params here
        hist = run(p)
        append_history_csv(out_file, p, hist, run_id=run_id)

    summarize_daily_all_runs("daily_all_runs.csv", "run_summaries.csv")
    print("Wrote run_summaries.csv")
    plot_run_summaries.run_plots()





