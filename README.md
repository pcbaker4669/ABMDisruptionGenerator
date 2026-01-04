# An Agent-Based Disruption Generator for Classrooms Using Lognormal Risk and Gamma–Poisson Counts

This repository contains the reference implementation for the disruption-only agent-based model (ABM) used in the paper:

**An Agent-Based Disruption Generator for Classrooms Using Lognormal Risk and Gamma–Poisson Counts**  
Peter C. Baker (George Mason University)

## Overview

The model simulates classroom disruption incidents as **event counts** in fixed **90-minute class periods**. Each student is an agent with a stable, latent disruption propensity (“risk”) drawn from a **lognormal distribution**, representing persistent between-student heterogeneity. For each class period, incidents are generated as counts using a **Gamma–Poisson mixture**, producing **overdispersed** (bursty) period totals with occasional spikes.

The model is intended as a **reusable core module** that can be integrated into larger classroom ABMs (e.g., models of instructional time loss, learning dynamics, and intervention experiments).

## Incident Definition (Measurement)

**Incident:** one discrete rule-violation (norm-violation) event observed during a **single 90-minute class period**, counted as a frequency measure.  
Incidents are **not severity-weighted** in this implementation.

## Key Outputs (Paper Artifacts)

The code produces:

- **Table 1**: baseline summary metrics (mean ± SD across replications)
- **Figure 1**: Lorenz curve of incidents across students (concentration)
- **Figure 2**: CCDF of class-period incident counts (tail/spikes)
- Tail probabilities: \(P(X \ge 10)\), \(P(X \ge 20)\), \(P(X \ge 30)\)

Optional (robustness / sensitivity):
- **Figure 3**: concentration vs risk dispersion (`risk_sigma`)
- **Figure 4**: overdispersion vs burstiness (`nb_k`)
- **Figure 5**: incident level vs baseline rate (`inc_base_rate`)

## Model Structure

- **Params** (dataclass): model configuration and parameters
- **Student**: agent with a stable latent `risk` and accumulated `incidents_total`
- **Model**:
  - initializes student risks (lognormal)
  - partitions students into fixed class rosters
  - simulates class periods over `n_days`
  - records class-period counts and summary metrics

## Parameters (Baseline)

Baseline configuration used in the paper:

- `n_students = 300`
- `class_size = 30` (10 fixed classes)
- `n_days = 90` (90 class periods)
- `risk_mu = -0.78`
- `risk_sigma = 1.6`  *(controls concentration across students)*
- `nb_k = 0.5`  *(controls burstiness / overdispersion via Gamma–Poisson mixture)*
- `inc_base_rate = 0.24`  *(controls overall incident level)*
- `at_risk_top_n = 3` *(defines the “at-risk” subgroup per class as top-N risk students)*

### Parameter roles (short)

- **`inc_base_rate`**: sets the overall level of incidents (mean incidents per class period)
- **`risk_sigma`**: increases/decreases concentration (higher values → more unequal incident contribution)
- **`nb_k`**: controls burstiness/overdispersion (lower values → heavier tails and more spikes)

## Requirements

- Python 3.10+ recommended
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
