# Implementing conformal prediction from the research paper

An implementation of Angelopoulos and Bates, *A Gentle Introduction to
Conformal Prediction and Distribution-Free Uncertainty Quantification*
(arXiv 2107.07511), and of conformalized quantile regression from Romano,
Patterson and Candes (NeurIPS 2019), tested on real ERCOT electricity demand.

The paper's central result is a finite-sample, distribution-free guarantee: for
any model and any data distribution, if the calibration and test points are
exchangeable, the conformal interval contains the true value with probability
at least 1 - alpha. I worked through the proof (the rank argument, why the
quantile is `ceil((n+1)(1-alpha))`, and why coverage for a fixed calibration set
is Beta distributed) until I could derive it myself. My write-up is in the math
section below. Then I implemented the methods without a conformal library,
checked each claim numerically, and tested what happens on real data when
exchangeability fails.

| Claim from the papers | How I checked it | Result |
|---|---|---|
| Coverage is at least 1 - alpha and below 1 - alpha + 1/(n+1) | 5,000 simulated trials with skewed scores | 0.898 against a 0.90 target, 0.5 standard errors off |
| Coverage for one calibration set follows Beta(k, n+1-k) | 2,000 calibration sets, 500 test points each | mean 0.902, sd 0.032 vs 0.901, 0.030 from the formula |
| The guarantee holds for any model, unlike a Gaussian interval | 200 random splits of the diabetes data | gradient boosting: 91.0% conformal vs 52.0% Gaussian |
| CQR adapts the interval width to difficulty | one split, binned by predicted uncertainty | widths 152 to 262 vs a constant 224 |
| All of it requires exchangeability | 43,848 hours of ERCOT demand, 2020-2024 | 90.0% shuffled; 89.1% in time order, with one month at 44.8% |

The scores, the quantile rule, the intervals and the CQR conformal step are my
own NumPy code; the base models are scikit-learn. Unit tests check the quantile
rule against examples I worked by hand.

`docs/index.html` is an interactive dashboard of the Texas results. It runs
offline, and it is also at
https://yerramsettianish0427.github.io/conformal-prediction-ercot/.

## The idea

The usual error bar assumes the model's errors follow a bell curve. Conformal
prediction drops that assumption. It measures how wrong the model was on data it
didn't train on and sizes the interval from that. The one requirement is that
new data is exchangeable with the calibration data. Time series usually aren't.

Example: a model predicts a used car sells for $12,000. On a few hundred cars it
never trained on, 90% of its errors were under $1,500. The interval is
$12,000 plus or minus $1,500, and if new cars behave like those, it contains the
true price about 90% of the time.

## The math

### Split conformal

Train a model `f` on one chunk of data. On a separate calibration set of `n`
points the model never saw, compute how wrong it was on each:

```
s_i = | y_i - f(x_i) |          for i = 1..n
```

Sort those scores. Let `k = ceil((n+1)(1-alpha))` and let `q_hat` be the k-th
smallest score. For a new point, the interval is:

```
[ f(x) - q_hat ,  f(x) + q_hat ]
```

and the guarantee is `P( y_new is inside ) >= 1 - alpha`.

### Why that's true

The new point is inside the interval exactly when its own score satisfies
`s_new <= q_hat`. If the `n` calibration scores and the new score are
exchangeable (roughly: shuffling them changes nothing), then the rank of `s_new`
among all `n+1` scores is equally likely to be 1, 2, ..., n+1. The event
`s_new <= q_hat` happens exactly when that rank is at most `k`, so:

```
P( s_new <= q_hat ) = k / (n+1)
```

Because `k >= (n+1)(1-alpha)`, that is at least `1 - alpha`. Because
`k < (n+1)(1-alpha) + 1`, it is also less than `1 - alpha + 1/(n+1)`, so the
guarantee can't be wildly conservative.

The `+1` is there because the new point is one of the things being ranked, so
there are `n+1` of them, not `n`. The ceiling is there because `k` has to be a
whole number, and rounding down would break the lower bound.

### Marginal, not conditional

That probability is averaged over everything, including the random calibration
set and every possible input `x`. It says nothing about any particular `x`. An
interval can be right 99% of the time on easy inputs and 60% on hard ones and
still average 90%.

### One calibration set is a draw, too

Fix a single calibration set and ask what coverage the resulting interval
actually delivers. That number varies from calibration set to calibration set,
and for continuous scores it follows `Beta(k, n+1-k)`. For `n = 100` and
`alpha = 0.1` that's `Beta(91, 10)`: mean 0.901, standard deviation about 0.03.
So a healthy calibration set can deliver 87% or 93%. In `coverage_simulation/`,
2,000 simulated calibration sets gave mean 0.902 and standard deviation 0.032.

### CQR

Split conformal uses one fixed width everywhere. CQR fits a low and a high
quantile model instead, then conformalizes the gap. The models are trained with
the pinball loss:

```
L_tau(y, q) = tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)
```

Its minimizer is the tau-th quantile of Y. Differentiate the expected loss with
respect to `q`:

```
d/dq E[L_tau] = -tau * P(Y > q) + (1 - tau) * P(Y < q) = F(q) - tau
```

Setting that to zero gives `F(q) = tau`, which is the definition of the tau-th
quantile. Fit `q_lo` at `tau = alpha/2` and `q_hi` at `tau = 1 - alpha/2`, then
score each calibration point with how far it fell outside that band:

```
E_i = max( q_lo(x_i) - y_i ,  y_i - q_hi(x_i) )
```

Take `Q` as the `ceil((n+1)(1-alpha))`-th smallest `E_i`, same rank rule as
before, and the interval is `[ q_lo(x) - Q ,  q_hi(x) + Q ]`. Since `q_lo` and
`q_hi` move with `x`, the interval gets wider where the model is unsure.

## Results

### Naive Gaussian vs conformal (scikit-learn diabetes data, 200 random splits)

Linear regression: split conformal 91.2%, naive Gaussian 88.1%. Gradient
boosting: split conformal 91.0%, naive Gaussian 52.0%. The naive interval uses
training residuals, which gradient boosting fits so tightly that the interval
comes out far too narrow. Split conformal never looks at training residuals.

### CQR

Split conformal gave every test point the same width, 224.3. CQR's widths ranged
from 152 to 262. Binned by predicted uncertainty, split conformal dipped to 89.7%
in the middle bin while CQR stayed near 93%. That was one split with about 30
points per bin, so it illustrates the mechanism and doesn't measure an effect.

### Texas, shuffled

On 43,848 hours of ERCOT demand, 50 random splits gave 90.0% coverage.

### Texas, in time order

Train on January to September 2020, calibrate on October to December 2020, then
predict 2021 through 2024 without recalibrating. Overall coverage is 89.1%:

```
2021  96.8%      Winter Storm Uri (Feb 10-20, 2021)   67.0%
2022  91.9%      Worst single month (Aug 2023)        44.8%
2023  85.2%
2024  82.6%
```

During Uri, all 87 missed hours were demand above the interval, none below. The
89.1% averages years that over-cover with years that under-cover.

There are two causes. The model was trained on 2020 only, so as Texas demand
grew its predictions stayed put: actual demand averaged 0.1 GW above prediction
in 2021 and 8.1 GW above in 2024. And the model has no weather input, so its
errors are much larger in hot months than in the autumn months the interval was
sized on. Average coverage by calendar month is 76% to 77% in June, August and
September and 96% to 97% in October to December.

### Monthly recalibration

My first guess was that the 2020 calibration went stale, so I recalibrated every
month on the trailing 90 days. That lifted 2024 from 82.6% to 87.5% but dropped
2021 from 96.8% to 87.8%, left Uri at 65.5%, and lowered overall coverage to
87.3%. A trailing window lags the seasons and can't anticipate a shock. A 30-day
window was worse (you can try it in the dashboard). This is a heuristic with no
guarantee. The principled version is adaptive conformal inference (Gibbs and
Candes, 2021), which I did not implement.

### Simulated returns

`returns_simulation/` applies the same chronological split to simulated GARCH(1,1)
daily returns: 92.7% coverage on calm days and 87.3% on turbulent ones. The data
is simulated.

## Limits

The Texas model uses only hour, weekday, month and a weekend flag, so these
numbers describe this model, not demand forecasting in general. The intervals
are 27 GW wide on a grid that runs from about 27 to 86 GW, because conformal
prediction guarantees coverage, not usefulness. It is one region and one model.

## Running it

```bash
pip install -r requirements.txt
bash reproduce.sh
```

This runs the simulations, the tests and every experiment, and rebuilds the
dashboard. The Texas random-split step takes a few minutes. To change the
dashboard, edit `texas_demand/dashboard_template.html` and run
`python texas_demand/make_dashboard.py`.

## Data and references

- EIA Cleaned Hourly Electricity Demand Data, 2025 release, `regions/TEX.csv`,
  by Ruggles, Farnham, Tong and Caldeira. Paper: https://doi.org/10.1038/s41597-020-0483-x,
  data: https://doi.org/10.5281/zenodo.3517196, CC BY 4.0. A copy of the Texas
  file is in `texas_demand/data/` so the repo runs offline.
- scikit-learn's built-in `diabetes` dataset for the split conformal and CQR
  experiments.
- Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction and
  Distribution-Free Uncertainty Quantification* (arXiv 2107.07511).
- Romano, Patterson and Candes, *Conformalized Quantile Regression* (NeurIPS 2019).
- Gibbs and Candes, *Adaptive Conformal Inference Under Distribution Shift* (2021).
