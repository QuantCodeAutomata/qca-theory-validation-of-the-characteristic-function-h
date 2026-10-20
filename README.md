# Characteristic-Function Half-Spread Estimator Theory Validation

Validates the analytical theory of the characteristic-function (CF) half-spread estimator under Roll's (1984) model and benchmarks it against Roll, augmented ECF-GMM, and Hasbrouck's Bayesian estimators.

## Structure

```
.
├── data/                     # Data loading scripts (no external data for Exp 1-2, 4)
├── src/
│   ├── theory.py             # CF estimator formulas (phi, bias, variance, mixture)
│   ├── roots.py              # Sign-change detection and Brent root finding for a*
│   ├── simulate.py           # Roll model simulation (Q_t, U_t)
│   ├── roll.py               # Roll (1984) estimator
│   ├── cf.py                 # Fixed-frequency CF estimator and CF-Roll mixture
│   ├── gmm.py                # Augmented ECF-GMM (two-step feasible GMM)
│   ├── hasbrouck.py          # Hasbrouck (2009) Bayesian MCMC estimator
│   ├── plotting.py           # Matplotlib visualizations
│   └── runtime.py            # Timing and system-info utilities
├── exp/
│   ├── exp1_theory.py        # Theory validation (no data needed)
│   ├── exp2_montecarlo.py    # Monte Carlo bias/RMSE comparison
│   ├── exp3_empirical.py     # Empirical cross-section (yfinance reconstruction)
│   └── exp4_runtime.py       # CPU runtime benchmark
├── tests/
│   ├── test_theory.py        # Theory and root-finding tests
│   └── test_estimators.py    # Estimator correctness tests
├── results/                  # Generated figures, CSV tables, RESULTS.md
└── README.md
```

## Experiments

### Experiment 1 — Theory Validation
Formula-driven validation of the CF estimator theory under Roll's model:
- CF identification identity: `1 - sqrt(phi(2a))/phi(a)^2 = tan^4(ac)`
- Transformed-bias curves `T(b)` vs frequency `a` (Figure 1 style)
- Bias-neutral frequency `a*` root finding and domain validation
- Appendix benchmark: `z = a*c ≈ 0.67855232` (ρ=1) and `≈ 0.69126736` (ρ=2)
- CF-Roll mixture weight `w*` and improvement condition `R > G`

### Experiment 2 — Monte Carlo Comparison
Synthetic benchmark under Roll's model (`c=0.02`, `n=252`, 1000 paths per SNR):
- Bias and RMSE curves vs SNR for Roll, CF, ECF-GMM, Hasbrouck
- Figure 2 style estimator distribution histograms
- Admissibility and convergence rates

### Experiment 3 — Empirical Cross-Section (Reconstruction)
Cross-sectional estimation for Indian equities 2023–2024.
Uses yfinance NSE data as CMIE Prowess substitute (labeled as reconstruction).
- Roll, augmented ECF-GMM, Hasbrouck per stock-year
- Figure 4 style comparison chart in log basis points

### Experiment 4 — Runtime Benchmark
CPU-only timing of all estimators on 1000 synthetic paths:
- Mean runtime per path with relative speed ratios
- Comparison to paper-reported values (Intel Xeon Gold 6530)

## Running Experiments

```bash
python exp/exp1_theory.py
python exp/exp2_montecarlo.py
python exp/exp3_empirical.py
python exp/exp4_runtime.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Formulas

**Roll model:** `Y_t = c(Q_t - Q_{t-1}) + U_t`,  `Q_t ∈ {-1,1}` iid, `U_t ~ N(0,V)`

**CF:** `phi(u) = exp(-Vu²/2) · cos²(cu)`

**Identification:** `1 - sqrt(phi(2a))/phi(a)² = tan⁴(ac)`

**CF estimator:** `c_CF(a) = (1/a) · arctan(d̂(a)^{1/4})` where `d̂ = 1 - sqrt(φ̂(2a))/φ̂(a)²`

## Dependencies

- numpy, scipy, pandas, matplotlib, seaborn
- yfinance (Experiment 3 only)

## Notes

- All custom formulas are labeled `# Custom — Context7 found no library equivalent`
- Experiment 3 uses yfinance (NSE data) as a reconstruction because CMIE Prowess is proprietary
- Hasbrouck MCMC runtime is very high (~seconds/path); Experiment 2 and 4 use reduced sample for practicality
