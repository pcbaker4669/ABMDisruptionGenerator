from dataclasses import dataclass
import numpy as np
import csv

# for analysis and error checking
# https://chatgpt.com/c/6945a0a5-b57c-8328-aaec-b7c5cfef2879

def write_history_csv(filename: str, history):
    # Writes one row per day so you can plot time series and compute summary stats later
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "day",
            "K_mean", "K_p10", "K_p50", "K_p90",
            "A_mean", "A_p10", "A_p50", "A_p90",
            "incidents_total",
            "time_lost_mean",
        ])
        for row in history:
            w.writerow(row)

def summarize_daily_csv(filename: str):
    rows = []
    with open(filename, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # convert numeric fields
            rows.append({
                "day": int(row["day"]),
                "K_mean": float(row["K_mean"]),
                "K_p10": float(row["K_p10"]),
                "A_mean": float(row["A_mean"]),
                "A_p10": float(row["A_p10"]),
                "incidents_total": int(row["incidents_total"]),
                "time_lost_mean": float(row["time_lost_mean"]),
            })

    first = rows[0]
    last = rows[-1]

    incidents_avg = sum(x["incidents_total"] for x in rows) / len(rows)
    time_lost_avg = sum(x["time_lost_mean"] for x in rows) / len(rows)

    print("Rows:", len(rows))
    print(f"Start: K_mean={first['K_mean']:.3f}, A_mean={first['A_mean']:.3f}")
    print(f"End:   K_mean={last['K_mean']:.3f}, K_p10={last['K_p10']:.3f}, "
          f"A_mean={last['A_mean']:.3f}, A_p10={last['A_p10']:.3f}")
    print(f"Avg incidents/day: {incidents_avg:.2f}")
    print(f"Avg time lost/day: {time_lost_avg:.3f}")

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
    n_days: int = 60
    seed: int = 1

    # Teacher skill distribution
    teacher_skill_mean: float = 0.80
    teacher_skill_sd: float = 0.10

    # Disruption model (new in V1)
    inc_c0: float = -2.2  # baseline (lower = fewer incidents)
    inc_c_class: float = 0.03  # effect of class size
    inc_c_B: float = 1.0  # effect of student behavior propensity

    base_loss: float = 0.03  # fraction of day lost per incident
    max_loss: float = 0.35  # cap on time lost per class per day

    # Agency (new in V2)
    agency_amp_min: float = 0.5     # A=0 -> multiplier 0.5
    agency_amp_max: float = 1.0     # A=1 -> multiplier 1.0

    beta_success: float = 0.010     # how much A increases on success
    gamma_failure: float = 0.006    # how much A decreases on failure
    success_eps: float = 1e-6       # what counts as "success" in dK

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
    # random shuffle of the student indices.
    # If n_students = 5, something like: array([3, 0, 4, 1, 2])
    order = rng.permutation(n_students)

    # "classes" becomes a list of arrays; each array holds the student IDs assigned
    # to one class for today (up to class_size_cap students per class).
    classes = []
    for i in range(0, n_students, class_size_cap):
        classes.append(order[i:i+class_size_cap])

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
    p = Params()
    hist = run(p)

    write_history_csv("daily.csv", hist)

    last = hist[-1]
    print("Done.")
    summarize_daily_csv("daily.csv")





