# model.py
from dataclasses import dataclass
import numpy as np
import math

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

    # Disruption process (Option A)
    inc_base_rate: float = 0.02   # baseline expected incidents per student per day (before modifiers)

    nb_k: float = 0.5           # dispersion; smaller => heavier tail,
                                # .5 - moderate burstiness
                                # .2 - strong burstiness

    # Latent risk distribution for students
    # risk_mu to hit your target average system load (total incidents/day).
    risk_mu: float = -0.78          # lognormal mean (in log space, originally 0)

    # Use risk_sigma to hit your target concentration (top 5% share ~.3 with value 1).
    # top 5% share ~.45 with value of 1.6
    risk_sigma: float = 1.6       # lognormal sigma (bigger => more concentration)


# ----------------------------
# Student agent
# ----------------------------
class Student:
    def __init__(self, sid: int, risk: float):
        self.sid = sid
        self.risk = float(risk)

        # bookkeeping
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
        self.history = []  # you decide what to record later


    def _init_students(self):

        # Latent disruption risk (positive, heavy-tailed)
        risk = self.rng.lognormal(mean=self.p.risk_mu, sigma=self.p.risk_sigma, size=self.p.n_students)

        return [Student(sid=i, risk=risk[i]) for i in range(self.p.n_students)]

    def _init_classes(self) -> list[np.ndarray]:
        # Partition students into fixed classes (rosters).
        order = self.rng.permutation(self.p.n_students)
        return [order[i:i + self.p.class_size] for i in range(0, self.p.n_students, self.p.class_size)]

    def step_day(self, day: int) -> None:
        total_incidents_today = 0

        # Process each class (fixed roster)
        class_totals = []
        for cls in self.classes:
            class_total = 0
            for sid in cls:
                s = self.students[int(sid)]

                # Mean incidents for this student today
                lam = self.p.inc_base_rate * s.risk

                # Gamma–Poisson mixture:
                #   lam_tilde ~ Gamma(k, scale=lam/k)
                #   k_i ~ Poisson(lam_tilde)
                k = max(self.p.nb_k, 1e-12)
                lam_tilde = self.rng.gamma(shape=k, scale=lam / k if lam > 0 else 0.0)
                k_i = int(self.rng.poisson(lam_tilde))

                s.incidents_total += k_i
                total_incidents_today += k_i
                class_total += k_i
            class_totals.append(class_total)
        self.history.append({
            "day": day,
            "incidents_total": total_incidents_today,
            "class_totals": class_totals,
        })

    def run(self):
        for day in range(1, self.p.n_days + 1):
            self.step_day(day)

        return self.history

def share_top(x, frac):
    x = np.asarray(x, dtype=float)
    s = x.sum()
    if s <= 0:
        return 0.0
    k = max(1, int(math.ceil(frac * len(x))))
    return np.sort(x)[::-1][:k].sum() / s

def summarize(model):
    daily = np.array([h["incidents_total"] for h in model.history], dtype=float)
    student_totals = np.array([s.incidents_total for s in model.students], dtype=float)

    print("Daily incidents: mean =", round(daily.mean(), 2),
          "var =", round(daily.var(ddof=1), 2),
          "var/mean =", round(daily.var(ddof=1)/max(daily.mean(), 1e-9), 2))

    print("Top 5% share (students):", round(share_top(student_totals, 0.05), 3))
    print("Top 1% share (students):", round(share_top(student_totals, 0.01), 3))
    print("Fraction of students with 0 incidents:", round((student_totals == 0).mean(), 3))

# ----------------------------
# Main
# ----------------------------
def main():
    p = Params()
    m = Model(p)
    hist = m.run()

    # minimal print so you know it ran
    print("Done.")
    print("Last day incidents_total =", hist[-1]["incidents_total"])
    summarize(m)


if __name__ == "__main__":
    main()
