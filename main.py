# https://chatgpt.com/c/69505735-7620-8329-b7e9-1458312497d8
# model.py
from dataclasses import dataclass
import numpy as np
import math
import csv
import matplotlib.pyplot as plt

# ----------------------------
# Parameters (model-specific)
# ----------------------------
@dataclass
class Params:
    seed: int = 1
    n_students: int = 300
    n_days: int = 90

    # Class structure
    class_size: int = 30  # fixed size (you can generalize later)

    # Disruption process, this scales class_mean (incident rate) linearly
    # 0.02 -> class_mean ~ 0.83,
    # if you want class_mean ~ 5 then .2 * (5/0.83) ~ 0.12
    # 0.24 -> class_mean ~ 10, 0.36 -> class_mean ~ .15
    inc_base_rate: float = 0.02   # baseline expected incidents per student per day (before modifiers)

    nb_k: float = 0.5           # dispersion; smaller => heavier tail, (originally 1)
                                # .5 - moderate burstiness
                                # .2 - strong burstiness

    # Latent risk distribution for students
    # risk_mu to hit your target average system load (total incidents/day).
    risk_mu: float = -0.78          # lognormal mean (in log space, originally 0)

    # Use risk_sigma to hit your target concentration (top 5% share ~.3 Originally 1).
    # top 5% share ~.45 with value of 1.6
    risk_sigma: float = 1.6       # lognormal sigma (bigger => more concentration)

    # for literature-style subgroup comparisons (e.g., "at-risk" students)
    at_risk_top_n: int = 3

# ----------------------------
# Student agent
# ----------------------------
class Student:
    def __init__(self, sid: int, risk: float):
        self.sid = sid
        self.risk = float(risk)
        self.incidents_total = 0


# ----------------------------
# Model
# ----------------------------
class Model:
    def __init__(self, p: Params):
        self.p = p
        self.rng = np.random.default_rng(p.seed)

        self.students = self._init_students()
        self.classes = self._init_classes()

        # map student -> class_id (useful for tables)
        self.class_of = {}
        for cid, cls in enumerate(self.classes):
            for sid in cls:
                self.class_of[int(sid)] = cid

        # define "at-risk" as top-N risk students per class (fixed for the run)
        self.at_risk_by_class = []
        for cls in self.classes:
            cls_ids = np.array([int(sid) for sid in cls])
            cls_risks = np.array([self.students[sid].risk for sid in cls_ids])
            top_idx = np.argsort(cls_risks)[::-1][: self.p.at_risk_top_n]
            self.at_risk_by_class.append(set(cls_ids[top_idx].tolist()))

        self.history = []             # one row per day (overall totals)
        self.class_day_records = []   # one row per class per day (literature-aligned)


    def _init_students(self):
        # Latent disruption risk (positive, heavy-tailed)
        risk = self.rng.lognormal(mean=self.p.risk_mu, sigma=self.p.risk_sigma, size=self.p.n_students)
        return [Student(sid=i, risk=risk[i]) for i in range(self.p.n_students)]

    def _init_classes(self) -> list[np.ndarray]:
        # Partition students into fixed classes (rosters).
        order = self.rng.permutation(self.p.n_students)
        return [order[i:i + self.p.class_size] for i in range(0, self.p.n_students, self.p.class_size)]

    def step_day(self, day: int):
        total_incidents_today = 0

        for cid, cls in enumerate(self.classes):
            class_total = 0
            at_risk_total = 0

            at_risk_set = self.at_risk_by_class[cid]

            for sid in cls:
                s = self.students[int(sid)]
                lam = self.p.inc_base_rate * s.risk

                k = max(self.p.nb_k, 1e-12)
                lam_tilde = self.rng.gamma(shape=k, scale=lam / k if lam > 0 else 0.0)
                k_i = int(self.rng.poisson(lam_tilde))

                s.incidents_total += k_i
                class_total += k_i
                total_incidents_today += k_i

                if int(sid) in at_risk_set:
                    at_risk_total += k_i

            self.class_day_records.append({
                "day": day,
                "class_id": cid,
                "class_size": len(cls),
                "incidents_class": class_total,
                "incidents_at_risk": at_risk_total,
                "incidents_nonrisk": class_total - at_risk_total,
                "at_risk_n": len(at_risk_set),
            })

        self.history.append({"day": day, "incidents_total": total_incidents_today})

    def run(self):
        for day in range(1, self.p.n_days + 1):
            self.step_day(day)
        return self.history

    @staticmethod
    def share_top(x, frac):
        x = np.asarray(x, dtype=float)
        s = x.sum()
        if s <= 0:
            return 0.0
        k = max(1, int(math.ceil(frac * len(x))))
        return np.sort(x)[::-1][:k].sum() / s

    def student_table(self):
        rows = []
        for s in self.students:
            rows.append({
                "sid": s.sid,
                "class_id": self.class_of[s.sid],
                "risk": s.risk,
                "incidents_total": s.incidents_total,
                "incidents_per_day": s.incidents_total / self.p.n_days,
            })
        return rows


    def summary(self):
        daily = np.array([h["incidents_total"] for h in self.history], dtype=float)

        student_totals = np.array([s.incidents_total for s in self.students], dtype=float)

        class_totals = np.array([r["incidents_class"] for r in self.class_day_records], dtype=float)
        at_risk_totals = np.array([r["incidents_at_risk"] for r in self.class_day_records], dtype=float)

        # per at-risk student per class-period (top N per class)
        at_risk_n = float(self.p.at_risk_top_n)
        at_risk_per_student = at_risk_totals / max(at_risk_n, 1.0)

        out = {
            "daily_mean": float(daily.mean()),
            "daily_var_mean": float(daily.var(ddof=1) / max(daily.mean(), 1e-12)),

            "class_mean": float(class_totals.mean()),
            "class_var_mean": float(class_totals.var(ddof=1) / max(class_totals.mean(), 1e-12)),
            "class_zero_frac": float((class_totals == 0).mean()),
            "class_q50": float(np.quantile(class_totals, 0.50)),
            "class_q90": float(np.quantile(class_totals, 0.90)),
            "class_q95": float(np.quantile(class_totals, 0.95)),
            "class_q99": float(np.quantile(class_totals, 0.99)),
            "class_max": float(class_totals.max()),

            "at_risk_class_mean": float(at_risk_totals.mean()),
            "at_risk_per_student_mean": float(at_risk_per_student.mean()),
            "at_risk_share_total": float(at_risk_totals.sum() / max(class_totals.sum(), 1e-12)),

            "top5_share_students": float(self.share_top(student_totals, 0.05)),
            "top1_share_students": float(self.share_top(student_totals, 0.01)),
            "student_zero_frac": float((student_totals == 0).mean()),
        }
        return out

def run_sweep():
    for rate in [0.12, 0.24, 0.36]:
        p = Params(inc_base_rate=rate)
        m = Model(p)
        m.run()
        s = m.summary()
        print("\ninc_base_rate =", rate)
        for k in ["class_mean","class_q50","class_q90","class_q95","class_q99","class_max",
                  "class_zero_frac","class_var_mean",
                  "at_risk_share_total","top5_share_students","student_zero_frac"]:
            print(k, "=", round(float(s[k]), 4))

def replicate_summaries(base_params, seeds=range(1, 51)):
    keys = [
        "class_mean","class_q50","class_q90","class_q95","class_q99","class_max",
        "class_zero_frac","class_var_mean",
        "at_risk_share_total","top5_share_students","student_zero_frac"
    ]
    rows = []
    for sd in seeds:
        p = Params(**{**base_params, "seed": sd})
        m = Model(p)
        m.run()
        s = m.summary()
        rows.append([float(s[k]) for k in keys])

    arr = np.array(rows, dtype=float)
    print("Replications:", len(seeds))
    for i, k in enumerate(keys):
        mean = arr[:, i].mean()
        sd = arr[:, i].std(ddof=1)
        print(f"{k}: mean={mean:.3f} sd={sd:.3f}")

def make_table1(base_params: dict, seeds=range(1, 51), out_csv="table1_baseline.csv"):
    """
    Runs replications over seeds and produces Table 1 (mean ± sd across runs).
    Saves per-run summaries and Table 1 to CSV.
    """
    keys = [
        "class_mean","class_q50","class_q90","class_q95","class_q99","class_max",
        "class_zero_frac","class_var_mean",
        "at_risk_share_total","top5_share_students","student_zero_frac"
    ]

    # Collect per-run rows
    rows = []
    for sd in seeds:
        p = Params(**{**base_params, "seed": sd})
        m = Model(p)
        m.run()
        s = m.summary()
        row = {k: float(s[k]) for k in keys}
        row["seed"] = sd
        rows.append(row)

    # Build numeric array for mean/sd
    X = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    means = X.mean(axis=0)
    sds = X.std(axis=0, ddof=1)

    # Print Table 1
    print("\nTABLE 1 (Baseline generator outputs across replications)")
    print(f"Replications: {len(list(seeds))}")
    for k, mu, sd in zip(keys, means, sds):
        print(f"{k}: mean={mu:.3f} sd={sd:.3f}")

    # Save per-run summaries
    per_run_csv = out_csv.replace(".csv", "_per_run.csv")
    with open(per_run_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed"] + keys)
        w.writeheader()
        w.writerows(rows)

    # Save Table 1 (mean/sd) as a simple CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "sd"])
        for k, mu, sd in zip(keys, means, sds):
            w.writerow([k, float(mu), float(sd)])

    print(f"\nWrote per-run summaries to: {per_run_csv}")
    print(f"Wrote Table 1 to: {out_csv}")

    return {k: (float(mu), float(sd)) for k, mu, sd in zip(keys, means, sds)}

import matplotlib.pyplot as plt

def share_top(x, frac):
    x = np.asarray(x, dtype=float)
    s = x.sum()
    if s <= 0:
        return 0.0
    k = max(1, int(math.ceil(frac * len(x))))
    return float(np.sort(x)[::-1][:k].sum() / s)

def lorenz_curve(values):
    v = np.asarray(values, dtype=float)
    v = np.maximum(v, 0.0)
    v_sorted = np.sort(v)
    cum = np.cumsum(v_sorted)
    total = cum[-1] if len(cum) else 0.0
    if total <= 0:
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        return x, y, 0.0

    x = np.concatenate([[0.0], np.arange(1, len(v_sorted) + 1) / len(v_sorted)])
    y = np.concatenate([[0.0], cum / total])

    area = np.trapezoid(y, x)
    gini = 1.0 - 2.0 * area
    return x, y, float(gini)

def plot_lorenz_from_model(
    base_params: dict,
    seeds=range(1, 51),
    out_png="fig1_lorenz_concentration.png",
    out_pdf="fig1_lorenz_concentration.pdf",
):
    """
    Runs the ABM across multiple seeds, pools student incident totals, and saves
    a journal-quality Lorenz curve figure (PNG + PDF).
    """
    seeds = list(seeds)

    pooled_totals = []
    top5_shares = []
    top1_shares = []

    for sd in seeds:
        p = Params(**{**base_params, "seed": sd})
        m = Model(p)
        m.run()
        totals = np.array([s.incidents_total for s in m.students], dtype=float)

        pooled_totals.append(totals)
        top5_shares.append(share_top(totals, 0.05))
        top1_shares.append(share_top(totals, 0.01))

    pooled_totals = np.concatenate(pooled_totals)
    x, y, gini = lorenz_curve(pooled_totals)

    top5_mean = float(np.mean(top5_shares))
    top1_mean = float(np.mean(top1_shares))

    # Journal-ish matplotlib defaults (no explicit colors)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
    })

    fig = plt.figure(figsize=(6.5, 5.0))
    ax = plt.gca()

    ax.plot(x, y, label="Lorenz curve (pooled student totals)")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Equality line")

    ax.set_title("Concentration of Incidents Across Students (Lorenz Curve)")
    ax.set_xlabel("Cumulative share of students")
    ax.set_ylabel("Cumulative share of incidents")

    txt = (
        f"Pooled over {len(seeds)} runs (N={len(pooled_totals)} student-runs)\n"
        f"Gini = {gini:.3f}\n"
        f"Mean top 5% share = {top5_mean:.3f}; mean top 1% share = {top1_mean:.3f}"
    )
    ax.text(0.05, 0.80, txt, transform=ax.transAxes)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


# ----------------------------
# Main
# ----------------------------
def main():
    # original run once
    # p = Params()
    # m = Model(p)
    # m.run()
    #
    # print("Summary:", m.summary())
    # print("Example class-day row:", m.class_day_records[0])
    # print("Example student row:", m.student_table()[0])

    # tuning
    # run_sweep()

    # multiple runs with different seeds
    base_params = dict(
        n_students=300, n_days=90, class_size=30,
        risk_mu=-0.78, risk_sigma=1.6, nb_k=0.5,
        inc_base_rate=0.24, at_risk_top_n=3
    )
    # replicate_summaries(base_params)
    make_table1(base_params, seeds=range(1, 51), out_csv="table1_baseline.csv")
    # Figure 1: Lorenz curve (student concentration)
    plot_lorenz_from_model(
        base_params,
        seeds=range(1, 51),
        out_png="fig1_lorenz_concentration.png",
        out_pdf="fig1_lorenz_concentration.pdf",
    )

if __name__ == "__main__":
    main()