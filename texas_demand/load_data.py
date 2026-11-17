"""
Loader for real ERCOT (Texas) hourly electricity demand, 2020-2024.

Source: EIA Cleaned Hourly Electricity Demand Data, 2025 release,
regions/TEX.csv (Ruggles et al., https://doi.org/10.1038/s41597-020-0483-x,
data at https://doi.org/10.5281/zenodo.3517196, CC BY 4.0).

Features are calendar only: hour, day of week, month, weekend flag. No
temperature, so the weather-driven part of demand ends up in the errors
the interval has to cover.
"""
import csv
import os
from datetime import datetime

import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "TEX.csv")


def load_ercot():
    """Returns (X, y, timestamps) sorted chronologically."""
    timestamps, demand = [], []
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(datetime.fromisoformat(row["date_time"]))
            demand.append(float(row["cleaned demand (MW)"]))

    X = np.array([
        [t.hour, t.weekday(), t.month, 1 if t.weekday() >= 5 else 0]
        for t in timestamps
    ], dtype=float)
    y = np.array(demand)
    return X, y, np.array(timestamps)


if __name__ == "__main__":
    X, y, ts = load_ercot()
    print(f"{len(y)} hourly observations, {ts[0]} to {ts[-1]}")
    print(f"demand range: {y.min():.0f} to {y.max():.0f} MW")
