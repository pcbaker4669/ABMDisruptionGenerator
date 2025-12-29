# model.py
from dataclasses import dataclass
import numpy as np


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
    class_size_slope: float = 0.0 # set 0 for now; add later if desired
    nb_k: float = 1.0             # dispersion; smaller => heavier tail

    # Latent risk distribution for students
    risk_mu: float = 0.0          # lognormal mean (in log space)
    risk_sigma: float = 1.0       # lognormal sigma (bigger => more concentration)

    # Learning (placeholder; you’ll add later)
    K0_a: float = 2.0
    K0_b: float = 3.0


# ----------------------------
# Student agent
# ----------------------------
class Student:
    def __init__(self, sid: int, K: float, risk: float):
        self.sid = sid
        self.K = float(K)
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
        self.history = []  # you decide what to record later

    def _init_students(self):
        # Mastery init (placeholder)
        K0 = self.rng.beta(self.p.K0_a, self.p.K0_b, self.p.n_students)

        # Latent disruption risk (positive, heavy-tailed)
        risk = self.rng.lognormal(mean=self.p.risk_mu, sigma=self.p.risk_sigma, size=self.p.n_students)

        return [Student(sid=i, K=K0[i], risk=risk[i]) for i in range(self.p.n_students)]

    def step_day(self, day: int):
        """Advance the model by one day."""
        # Shuffle roster each day (keep for now; you can later fix rosters)
        order = self.rng.permutation(self.p.n_students)

        total_incidents_today = 0

        # Process each class
        for start in range(0, self.p.n_students, self.p.class_size):
            cls = order[start:start + self.p.class_size]
            class_size = len(cls)
            if class_size == 0:
                continue

            # Simple class-size factor (optional; keep it neutral at first)
            class_factor = np.exp(self.p.class_size_slope * class_size)

            for sid in cls:
                s = self.students[int(sid)]

                # Mean incidents for this student today
                lam = self.p.inc_base_rate * class_factor * s.risk

                # NegBin via Gamma–Poisson mixture (overdispersed counts)
                lam_tilde = self.rng.gamma(shape=self.p.nb_k, scale=lam / max(self.p.nb_k, 1e-9))
                k_i = int(self.rng.poisson(lam_tilde))

                s.incidents_total += k_i
                total_incidents_today += k_i

        # Record minimal daily output (add more later)
        self.history.append({"day": day, "incidents_total": total_incidents_today})

    def run(self):
        for day in range(1, self.p.n_days + 1):
            self.step_day(day)
        return self.history


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


if __name__ == "__main__":
    main()
