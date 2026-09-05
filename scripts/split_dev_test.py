"""
Splits ground_truth.csv into dev_set and test_set, stratified by case_type,
so both sets get a fair mix of easy/hard cases rather than the test set
getting all the easy ones (or all the hard ones) by chance.

This script only touches ground_truth.csv - it does NOT touch or filter
transactions/settlements/refunds/fees, since the engine works off the
full pool of visible records regardless of which cases are "dev" or "test".
The split only matters when we grade afterward.
"""

import csv
import random
import os
from collections import defaultdict

random.seed(7)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_PATH = PROJECT_ROOT + "/data/ground_truth/ground_truth.csv"

with open(GT_PATH) as f:
    rows = list(csv.DictReader(f))

by_type = defaultdict(list)
for r in rows:
    by_type[r["case_type"]].append(r)

dev_rows, test_rows = [], []
for case_type, group in by_type.items():
    random.shuffle(group)
    split_point = max(1, round(len(group) * 0.7))
    dev_rows.extend(group[:split_point])
    test_rows.extend(group[split_point:])

fieldnames = list(rows[0].keys())

def write(path, data):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ["split"])
        writer.writeheader()
        for r in data:
            r2 = dict(r)
            r2["split"] = "dev" if data is dev_rows else "test"
            writer.writerow(r2)

write(PROJECT_ROOT + "/data/ground_truth/ground_truth_dev.csv", dev_rows)
write(PROJECT_ROOT + "/data/ground_truth/ground_truth_test.csv", test_rows)

print(f"Dev set : {len(dev_rows)} cases")
print(f"Test set: {len(test_rows)} cases")
print("\nPer-type split:")
for case_type, group in by_type.items():
    n_dev = sum(1 for r in dev_rows if r["case_type"] == case_type)
    n_test = sum(1 for r in test_rows if r["case_type"] == case_type)
    print(f"  {case_type}: dev={n_dev}, test={n_test}")
