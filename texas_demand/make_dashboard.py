"""
Builds docs/index.html from dashboard_template.html.

Trains the same model as the experiments and embeds the calibration errors
and 2021-2024 predictions/actuals as JSON. The quantile and coverage math runs
in the browser, so the page works offline. Edit the template, not
docs/index.html, since this script overwrites it.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "split_conformal", "src"))
sys.path.insert(0, HERE)

import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor

from conformal import compute_scores
from load_data import load_ercot

X, y, ts = load_ercot()
train_mask = ts < datetime(2020, 10, 1)
cal_mask = (ts >= datetime(2020, 10, 1)) & (ts < datetime(2021, 1, 1))
test_mask = ts >= datetime(2021, 1, 1)

model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=0)
model.fit(X[train_mask], y[train_mask])

cal_err = compute_scores(y[cal_mask], model.predict(X[cal_mask]))
pred = model.predict(X[test_mask])
actual = y[test_mask]
ts_test = ts[test_mask]


def clean(v):
    # 1 decimal, ints where possible, keeps the file small
    v = round(float(v), 1)
    return int(v) if v == int(v) else v


# month boundaries (index into the test arrays)
months, start = [], 0
keys = [(t.year, t.month) for t in ts_test]
for i in range(1, len(keys) + 1):
    if i == len(keys) or keys[i] != keys[start]:
        months.append({"y": keys[start][0], "m": keys[start][1], "start": start, "end": i})
        start = i

uri_idx = [i for i, t in enumerate(ts_test) if t.year == 2021 and t.month == 2 and 10 <= t.day <= 20]

data = {
    "calErr": [clean(v) for v in cal_err],
    "actual": [clean(v) for v in actual],
    "pred": [round(float(v), 1) for v in pred],
    "months": months,
    "uri": [uri_idx[0], uri_idx[-1]],
}
payload = json.dumps(data, separators=(",", ":"))

template = open(os.path.join(HERE, "dashboard_template.html"), encoding="utf-8").read()
assert "__DATA_JSON__" in template, "placeholder missing from template"
page = template.replace("__DATA_JSON__", payload)

# template is split into head/body at this marker
head, body = page.split("<!--HEAD-END-->", 1)
html = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    + head + "\n</head>\n<body>\n" + body + "\n</body>\n</html>\n"
)

out = os.path.join(HERE, "..", "docs", "index.html")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"wrote docs/index.html  ({len(html) / 1024:.0f} KB, {len(actual)} test hours, {len(months)} months)")
