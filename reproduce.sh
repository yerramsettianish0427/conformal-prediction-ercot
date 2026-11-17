#!/bin/bash
# Runs everything: simulations, tests, experiments, dashboard build.
set -e

echo "=== Coverage guarantee and Beta distribution, by simulation ==="
cd coverage_simulation && python3 simulate.py && python3 beta_check.py && cd ..

echo ""
echo "=== Split conformal vs naive Gaussian (diabetes) ==="
cd split_conformal && python3 -m pytest tests/ -q && python3 experiment.py && cd ..

echo ""
echo "=== CQR vs split conformal ==="
cd cqr && python3 -m pytest tests/ -q && python3 experiment.py && cd ..

echo ""
echo "=== Texas demand, shuffled vs deployed (a few minutes) ==="
cd texas_demand
python3 experiment_random_split.py
python3 experiment_chronological.py
python3 experiment_rolling_recalibration.py
python3 make_dashboard.py
cd ..

echo ""
echo "=== Side experiment: simulated returns with volatility clustering ==="
cd returns_simulation && python3 simulate_returns.py && python3 experiment_stock_break.py && cd ..

echo ""
echo "=== Done. Open docs/index.html to see the dashboard. ==="
