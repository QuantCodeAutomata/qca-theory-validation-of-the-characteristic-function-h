# AGENTS.md — Repository Knowledge Base

## Project Purpose
Theory validation and Monte Carlo benchmarking of the characteristic-function (CF)
half-spread estimator under Roll's (1984) market microstructure model.

## Repository Layout
- `src/`         — Core modules (no external data dependencies for Exp 1, 2, 4)
- `exp/`         — Runnable experiment scripts (exp1_theory, exp2_montecarlo, exp3_empirical, exp4_runtime)
- `tests/`       — pytest unit and regression tests
- `results/`     — Generated PNG figures, CSV tables, RESULTS.md
- `data/`        — Data loading scripts (Exp 3 uses yfinance as CMIE Prowess reconstruction)

## Key Model Equations
- Roll model: `Y_t = c*(Q_t - Q_{t-1}) + U_t`, Q_t ∈ {-1,+1} iid, U_t ~ N(0,V)
- CF:         `phi(u) = exp(-V*u²/2) * cos²(c*u)`
- Identity:   `1 - sqrt(phi(2a))/phi(a)² = tan⁴(ac)`
- Admissible domain: `0 < a*c < pi/4`
- CF estimator: `c_hat = (1/a)*arctan(d_hat^(1/4))`, `d_hat = 1 - sqrt(phi_hat(2a))/phi_hat(a)²`

## Appendix Benchmark Values (for regression tests)
- rho=1 (V=c²):  z* = a*·c ≈ 0.67855232
- rho=2 (V=2c²): z* = a*·c ≈ 0.69126736

## Module Responsibilities
| Module           | Purpose |
|------------------|---------|
| src/theory.py    | phi, bias_coeff, variance_coeff, roll_asymptotic, cf_roll_mixture |
| src/roots.py     | build_frequency_grid, find_interior_root (grid + Brent) |
| src/simulate.py  | simulate_roll_path, simulate_roll_paths, validate_moments |
| src/roll.py      | roll_estimator, roll_estimator_batch |
| src/cf.py        | cf_estimator_single, cf_estimator_batch, cf_roll_mixture_estimator |
| src/gmm.py       | ecf_gmm_augmented (two-step feasible GMM, DEFAULT_FREQS=[5,10,20,40]) |
| src/hasbrouck.py | hasbrouck_mcmc (Gibbs: Q → c → V, burn_in=1000, n_draws=1500) |
| src/plotting.py  | All matplotlib figures (Agg backend) |
| src/runtime.py   | get_system_info, time_estimator, summarize_timings |

## Important Design Decisions
- All formulas in `src/theory.py` are custom (no library equivalent via Context7).
- GMM frequencies [5, 10, 20, 40] are a reconstruction (paper does not fully specify).
- Hasbrouck priors are reconstructed from Hasbrouck (2009) Section C.
- Exp 3 uses yfinance NSE data as a CMIE Prowess substitute (labeled as reconstruction).
- Hasbrouck is slow (~1-10s/path); Exp 2 uses N_HAS=50 paths for Hasbrouck.
- Use float64 throughout.

## Running Experiments
```bash
python exp/exp1_theory.py     # ~30s (formula-only, fast)
python exp/exp2_montecarlo.py # ~30-90min (1000 paths, Hasbrouck subsampled)
python exp/exp3_empirical.py  # ~10min (yfinance + Hasbrouck)
python exp/exp4_runtime.py    # ~15min (benchmarking)
pytest tests/ -v              # ~5min
```

## Testing Conventions
- test_theory.py covers: identity, bias sign pattern, root finding, appendix benchmarks
- test_estimators.py covers: Roll, CF, GMM, Hasbrouck output validity
- Tolerance for appendix z* benchmarks: 1e-4 (Brent converges to 1e-14)

## Known Reconstruction Choices
1. Exp 3: yfinance instead of CMIE Prowess (labeled)
2. GMM frequencies: [5, 10, 20, 40] (paper says ~"6 frequencies" or "J=4")
3. Hasbrouck priors: mu0=0.02, tau0_sq=0.01, a0=3, b0=0.0001 (Section C reconstruction)
4. Exp 2 Hasbrouck: 50-path subsample + extrapolation (runtime constraint)
