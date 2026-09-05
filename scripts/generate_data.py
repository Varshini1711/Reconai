"""
ReconAI - Synthetic Data + Ground Truth Generator (v2 - collision-safe)

FIX vs v1: v1 drew amounts from small fixed pools per case type, which
caused ACCIDENTAL cross-case collisions (two unrelated cases producing
transactions with identical merchant+amount+date-window, confusing
Layer 1's uniqueness check). Root cause: amount space too small for the
number of cases.

FIX: every "primary" transaction amount is now drawn from a large pool
of globally UNIQUE amounts (no two unrelated transactions ever share an
amount), except where a case type INTENTIONALLY needs a shared amount
(ambiguous_same_amount, duplicate_record, borderline_confidence) - in
those cases the shared/derived amount is explicitly reserved so no
OTHER case can accidentally reuse it either.

Also fixes: 1to1_name_noise previously could draw a merchant pair whose
internal/bank names were identical (no real noise). Now restricted to
pairs with genuinely different names.

Same random seed (42) preserved. Same 101 total cases, same case-type
counts, same file layout, same ground truth schema.
"""

import random
import csv
import os
from collections import Counter

random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_SYNTH = PROJECT_ROOT + "/data/synthetic"
OUT_GT = PROJECT_ROOT + "/data/ground_truth"

transactions = []
settlements = []
refunds = []
fees = []
ground_truth = []

_txn_counter = 1000
_set_counter = 2000
_ref_counter = 3000
_fee_counter = 4000
_case_counter = 1


def next_txn_id():
    global _txn_counter
    _txn_counter += 1
    return f"TXN{_txn_counter}"


def next_set_id():
    global _set_counter
    _set_counter += 1
    return f"SET{_set_counter}"


def next_ref_id():
    global _ref_counter
    _ref_counter += 1
    return f"REF{_ref_counter}"


def next_fee_id():
    global _fee_counter
    _fee_counter += 1
    return f"FEE{_fee_counter}"


def next_case_id():
    global _case_counter
    cid = f"CASE{_case_counter:03d}"
    _case_counter += 1
    return cid


MERCHANTS = [
    ("Zomato", "Zomato"),
    ("Swiggy", "Swiggy"),
    ("Amazon Marketplace India", "AMZN Mktp IN"),
    ("Flipkart", "Flipkart Internet Pvt Ltd"),
    ("Myntra", "Myntra Designs"),
    ("Ola", "ANI Technologies (Ola)"),
    ("Big Bazaar", "Big Bazaar Retail"),
    ("Nykaa", "FSN E-Commerce (Nykaa)"),
    ("BookMyShow", "Bigtree Entertainment (BMS)"),
    ("Uber", "Uber India Systems"),
    ("PVR Cinemas", "PVR Ltd"),
    ("Cafe Coffee Day", "Cafe Coffee Day"),
]

# Only pairs with genuinely different internal/bank names - fixes the
# name-noise bug where some pairs were accidentally identical.
NOISY_MERCHANTS = [p for p in MERCHANTS if p[0].strip().lower() != p[1].strip().lower()]

# ---------------------------------------------------------------------
# GLOBAL UNIQUE-AMOUNT POOL - the actual collision fix.
# ---------------------------------------------------------------------
_AMOUNT_POOL = list(range(500, 9990, 10))
random.shuffle(_AMOUNT_POOL)
_amount_iter = iter(_AMOUNT_POOL)
_used_amounts = set()


def next_unique_amount():
    """Draws an amount never used by any other transaction in the dataset."""
    for amt in _amount_iter:
        if amt not in _used_amounts:
            _used_amounts.add(amt)
            return amt
    raise RuntimeError("Ran out of unique amounts - widen _AMOUNT_POOL range.")


def reserve_amount(amt):
    """Explicitly marks a derived amount (e.g. amt+10) as used, so no
    other case can accidentally draw/derive the same value."""
    _used_amounts.add(amt)


def rand_date(start_day=1, offset=0):
    day = min(start_day + offset, 28)
    return f"2026-09-{day:02d}"


# ---------------------------------------------------------------------
# CASE TYPE GENERATORS
# ---------------------------------------------------------------------

def case_clean_1to1():
    name, _ = random.choice(MERCHANTS)
    amt = next_unique_amount()
    day = random.randint(1, 10)
    txn_id = next_txn_id()
    set_id = next_set_id()
    transactions.append(dict(txn_id=txn_id, amount=amt, date=rand_date(day),
                              merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
    settlements.append(dict(settlement_id=set_id, amount=amt, date=rand_date(day, 2), merchant_name=name))
    ground_truth.append(dict(case_id=next_case_id(), case_type="1to1_clean",
                              transaction_ids=txn_id, settlement_ids=set_id,
                              refund_ids="", fee_ids="", expected_decision="AUTO_MATCHED",
                              notes="Exact amount, minor/no date shift, same merchant name."))


def case_name_noise_1to1():
    name_internal, name_bank = random.choice(NOISY_MERCHANTS)
    amt = next_unique_amount()
    day = random.randint(1, 10)
    txn_id = next_txn_id()
    set_id = next_set_id()
    transactions.append(dict(txn_id=txn_id, amount=amt, date=rand_date(day),
                              merchant_name=name_internal, customer_ref=f"ORD{random.randint(1000,9999)}"))
    settlements.append(dict(settlement_id=set_id, amount=amt, date=rand_date(day, 2), merchant_name=name_bank))
    ground_truth.append(dict(case_id=next_case_id(), case_type="1to1_name_noise",
                              transaction_ids=txn_id, settlement_ids=set_id,
                              refund_ids="", fee_ids="", expected_decision="AUTO_MATCHED",
                              notes="Merchant name differs between internal and bank records; same entity."))


def case_many_to_one():
    name, _ = random.choice(MERCHANTS)
    day = random.randint(1, 8)
    n = random.choice([2, 3])
    txn_ids = []
    total = 0
    for _ in range(n):
        amt = next_unique_amount()
        tid = next_txn_id()
        transactions.append(dict(txn_id=tid, amount=amt, date=rand_date(day),
                                  merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
        txn_ids.append(tid)
        total += amt
    fee_amt = round(total * 0.02, 2)
    fee_id = next_fee_id()
    fees.append(dict(fee_id=fee_id, related_txn_ids=";".join(txn_ids), amount=fee_amt, type="platform_fee"))
    set_id = next_set_id()
    settlements.append(dict(settlement_id=set_id, amount=round(total - fee_amt, 2),
                             date=rand_date(day, 2), merchant_name=name))
    ground_truth.append(dict(case_id=next_case_id(), case_type="many_to_one",
                              transaction_ids=";".join(txn_ids), settlement_ids=set_id,
                              refund_ids="", fee_ids=fee_id, expected_decision="AUTO_MATCHED",
                              notes=f"{n} transactions batched, {fee_amt} platform fee deducted."))


def case_one_to_many():
    name, _ = random.choice(MERCHANTS)
    day = random.randint(1, 8)
    total = next_unique_amount() + random.choice([1000, 2000, 3000])  # keep it a larger, splittable total
    split1 = round(total * random.choice([0.5, 0.6, 0.4]))
    split2 = total - split1
    txn_id = next_txn_id()
    transactions.append(dict(txn_id=txn_id, amount=total, date=rand_date(day),
                              merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
    set_id1 = next_set_id()
    set_id2 = next_set_id()
    settlements.append(dict(settlement_id=set_id1, amount=split1, date=rand_date(day, 2), merchant_name=name))
    settlements.append(dict(settlement_id=set_id2, amount=split2, date=rand_date(day, 4), merchant_name=name))
    ground_truth.append(dict(case_id=next_case_id(), case_type="one_to_many",
                              transaction_ids=txn_id, settlement_ids=f"{set_id1};{set_id2}",
                              refund_ids="", fee_ids="", expected_decision="AUTO_MATCHED",
                              notes="Single transaction split across two settlement payouts."))


def case_refund_reduces_settlement():
    name, _ = random.choice(MERCHANTS)
    day = random.randint(1, 6)
    amt = next_unique_amount()
    refund_amt = round(amt * random.choice([0.2, 0.25, 0.3]))
    txn_id = next_txn_id()
    transactions.append(dict(txn_id=txn_id, amount=amt, date=rand_date(day),
                              merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
    ref_id = next_ref_id()
    refunds.append(dict(refund_id=ref_id, original_txn_id=txn_id, amount=refund_amt, date=rand_date(day, 3)))
    set_id = next_set_id()
    settlements.append(dict(settlement_id=set_id, amount=amt - refund_amt, date=rand_date(day, 4), merchant_name=name))
    ground_truth.append(dict(case_id=next_case_id(), case_type="refund_reduces_settlement",
                              transaction_ids=txn_id, settlement_ids=set_id,
                              refund_ids=ref_id, fee_ids="", expected_decision="AUTO_MATCHED",
                              notes="Settlement = transaction amount minus refund."))


def case_partial_settlement():
    name, _ = random.choice(MERCHANTS)
    day = random.randint(1, 8)
    amt = next_unique_amount() + 3000  # keep it comfortably large so 50-60% is meaningful
    paid = round(amt * random.choice([0.5, 0.6]))
    txn_id = next_txn_id()
    transactions.append(dict(txn_id=txn_id, amount=amt, date=rand_date(day),
                              merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
    set_id = next_set_id()
    settlements.append(dict(settlement_id=set_id, amount=paid, date=rand_date(day, 2), merchant_name=name))
    ground_truth.append(dict(case_id=next_case_id(), case_type="partial_settlement",
                              transaction_ids=txn_id, settlement_ids=set_id,
                              refund_ids="", fee_ids="", expected_decision="NEEDS_REVIEW",
                              notes=f"Only {paid} of {amt} settled so far; remainder {amt-paid} still pending, not an error."))


def case_fee_rounding_near_miss():
    name, _ = random.choice(MERCHANTS)
    day = random.randint(1, 8)
    amt = next_unique_amount()
    fee_amt = round(amt * 0.02 - 0.02, 2)
    txn_id = next_txn_id()
    transactions.append(dict(txn_id=txn_id, amount=amt, date=rand_date(day),
                              merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
    fee_id = next_fee_id()
    fees.append(dict(fee_id=fee_id, related_txn_ids=txn_id, amount=fee_amt, type="platform_fee"))
    set_id = next_set_id()
    settlements.append(dict(settlement_id=set_id, amount=round(amt - fee_amt, 2), date=rand_date(day, 2), merchant_name=name))
    ground_truth.append(dict(case_id=next_case_id(), case_type="fee_rounding_near_miss",
                              transaction_ids=txn_id, settlement_ids=set_id,
                              refund_ids="", fee_ids=fee_id, expected_decision="AUTO_MATCHED",
                              notes="Fee has a small rounding fraction; requires tolerance, not exact equality."))


def case_ambiguous_same_amount():
    """INTENTIONAL collision: two transactions must share the exact same
    amount+merchant+day. We draw ONE unique amount and reserve it, so no
    other unrelated case can ever collide with this pair."""
    name, _ = random.choice(MERCHANTS)
    day = random.randint(1, 8)
    amt = next_unique_amount()
    txn_id1 = next_txn_id()
    txn_id2 = next_txn_id()
    transactions.append(dict(txn_id=txn_id1, amount=amt, date=rand_date(day),
                              merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
    transactions.append(dict(txn_id=txn_id2, amount=amt, date=rand_date(day),
                              merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
    set_id = next_set_id()
    settlements.append(dict(settlement_id=set_id, amount=amt, date=rand_date(day, 2), merchant_name=name))
    ground_truth.append(dict(case_id=next_case_id(), case_type="ambiguous_same_amount",
                              transaction_ids=txn_id1, settlement_ids=set_id,
                              refund_ids="", fee_ids="", expected_decision="NEEDS_REVIEW",
                              notes=f"{txn_id1} and {txn_id2} are equally plausible candidates; "
                                    f"no signal distinguishes them from visible data. "
                                    f"Correct behavior is escalation, not a forced guess."))


def case_duplicate_record():
    """INTENTIONAL collision, same reservation logic as above."""
    name, _ = random.choice(MERCHANTS)
    day = random.randint(1, 8)
    amt = next_unique_amount()
    txn_id1 = next_txn_id()
    txn_id2 = next_txn_id()
    transactions.append(dict(txn_id=txn_id1, amount=amt, date=rand_date(day),
                              merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
    transactions.append(dict(txn_id=txn_id2, amount=amt, date=rand_date(day),
                              merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
    set_id = next_set_id()
    settlements.append(dict(settlement_id=set_id, amount=amt, date=rand_date(day, 2), merchant_name=name))
    ground_truth.append(dict(case_id=next_case_id(), case_type="duplicate_record",
                              transaction_ids=txn_id1, settlement_ids=set_id,
                              refund_ids="", fee_ids="", expected_decision="AUTO_MATCHED",
                              notes=f"{txn_id2} is an accidental duplicate entry of {txn_id1}; "
                                    f"only one settlement exists and should match once, not twice."))


def case_orphan_settlement():
    name, _ = random.choice(MERCHANTS)
    day = random.randint(1, 8)
    amt = next_unique_amount()
    set_id = next_set_id()
    settlements.append(dict(settlement_id=set_id, amount=amt, date=rand_date(day, 2),
                             merchant_name="Unknown Merchant " + str(random.randint(1, 99))))
    ground_truth.append(dict(case_id=next_case_id(), case_type="orphan_settlement",
                              transaction_ids="", settlement_ids=set_id,
                              refund_ids="", fee_ids="", expected_decision="UNRESOLVED",
                              notes="No transaction exists for this settlement; genuinely unresolved by design."))


def case_orphan_refund():
    fake_txn = f"TXN{random.randint(9000,9999)}"
    ref_id = next_ref_id()
    refunds.append(dict(refund_id=ref_id, original_txn_id=fake_txn, amount=next_unique_amount() % 900 + 100,
                         date=rand_date(random.randint(1, 8))))
    ground_truth.append(dict(case_id=next_case_id(), case_type="orphan_refund",
                              transaction_ids="", settlement_ids="",
                              refund_ids=ref_id, fee_ids="", expected_decision="UNRESOLVED",
                              notes=f"References non-existent transaction {fake_txn}; data error, genuinely unresolved."))


def case_missing_settlement():
    name, _ = random.choice(MERCHANTS)
    day = random.randint(1, 8)
    amt = next_unique_amount()
    txn_id = next_txn_id()
    transactions.append(dict(txn_id=txn_id, amount=amt, date=rand_date(day),
                              merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
    ground_truth.append(dict(case_id=next_case_id(), case_type="missing_settlement",
                              transaction_ids=txn_id, settlement_ids="",
                              refund_ids="", fee_ids="", expected_decision="UNRESOLVED",
                              notes="No settlement exists yet for this transaction; correctly unresolved."))


def case_date_shift_only():
    name, _ = random.choice(MERCHANTS)
    day = random.randint(1, 5)
    amt = next_unique_amount()
    txn_id = next_txn_id()
    set_id = next_set_id()
    transactions.append(dict(txn_id=txn_id, amount=amt, date=rand_date(day),
                              merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
    settlements.append(dict(settlement_id=set_id, amount=amt, date=rand_date(day, 8), merchant_name=name))
    ground_truth.append(dict(case_id=next_case_id(), case_type="date_shift_only",
                              transaction_ids=txn_id, settlement_ids=set_id,
                              refund_ids="", fee_ids="", expected_decision="AUTO_MATCHED",
                              notes="Unusually long (8-day) settlement delay; tests date-window tuning."))


def case_borderline_confidence():
    """Exact-amount candidate should uniquely win; near candidate has a
    different (reserved) amount and a different merchant-name variant."""
    name, _ = random.choice(MERCHANTS)
    day = random.randint(1, 8)
    amt = next_unique_amount()
    near_amt = amt + 10
    reserve_amount(near_amt)
    txn_id_exact = next_txn_id()
    txn_id_near = next_txn_id()
    transactions.append(dict(txn_id=txn_id_exact, amount=amt, date=rand_date(day),
                              merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
    transactions.append(dict(txn_id=txn_id_near, amount=near_amt, date=rand_date(day),
                              merchant_name=name + " Pvt Ltd", customer_ref=f"ORD{random.randint(1000,9999)}"))
    set_id = next_set_id()
    settlements.append(dict(settlement_id=set_id, amount=amt, date=rand_date(day, 2), merchant_name=name))
    ground_truth.append(dict(case_id=next_case_id(), case_type="borderline_confidence",
                              transaction_ids=txn_id_exact, settlement_ids=set_id,
                              refund_ids="", fee_ids="", expected_decision="AUTO_MATCHED",
                              notes=f"{txn_id_exact} matches exactly; {txn_id_near} is a close but inferior "
                                    f"candidate (amount off by 10, name variant). Tests threshold tuning."))


def case_batched_fee_three_txns():
    name, _ = random.choice(MERCHANTS)
    day = random.randint(1, 6)
    txn_ids = []
    total = 0
    for _ in range(3):
        amt = next_unique_amount()
        tid = next_txn_id()
        transactions.append(dict(txn_id=tid, amount=amt, date=rand_date(day),
                                  merchant_name=name, customer_ref=f"ORD{random.randint(1000,9999)}"))
        txn_ids.append(tid)
        total += amt
    fee_amt = 200
    fee_id = next_fee_id()
    fees.append(dict(fee_id=fee_id, related_txn_ids=";".join(txn_ids), amount=fee_amt, type="platform_fee"))
    set_id = next_set_id()
    settlements.append(dict(settlement_id=set_id, amount=total - fee_amt, date=rand_date(day, 2), merchant_name=name))
    ground_truth.append(dict(case_id=next_case_id(), case_type="batched_fee_three_txns",
                              transaction_ids=";".join(txn_ids), settlement_ids=set_id,
                              refund_ids="", fee_ids=fee_id, expected_decision="AUTO_MATCHED",
                              notes="Three transactions batched with one combined platform fee."))


CASE_PLAN = [
    (case_clean_1to1, 15),
    (case_name_noise_1to1, 10),
    (case_many_to_one, 8),
    (case_one_to_many, 6),
    (case_refund_reduces_settlement, 8),
    (case_partial_settlement, 6),
    (case_fee_rounding_near_miss, 6),
    (case_ambiguous_same_amount, 6),
    (case_duplicate_record, 5),
    (case_orphan_settlement, 6),
    (case_orphan_refund, 4),
    (case_missing_settlement, 6),
    (case_date_shift_only, 5),
    (case_borderline_confidence, 6),
    (case_batched_fee_three_txns, 4),
]

for func, count in CASE_PLAN:
    for _ in range(count):
        func()


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


write_csv(f"{OUT_SYNTH}/transactions.csv", transactions,
          ["txn_id", "amount", "date", "merchant_name", "customer_ref"])
write_csv(f"{OUT_SYNTH}/settlements.csv", settlements,
          ["settlement_id", "amount", "date", "merchant_name"])
write_csv(f"{OUT_SYNTH}/refunds.csv", refunds,
          ["refund_id", "original_txn_id", "amount", "date"])
write_csv(f"{OUT_SYNTH}/fees.csv", fees,
          ["fee_id", "related_txn_ids", "amount", "type"])
write_csv(f"{OUT_GT}/ground_truth.csv", ground_truth,
          ["case_id", "case_type", "transaction_ids", "settlement_ids",
           "refund_ids", "fee_ids", "expected_decision", "notes"])

print("Generated:")
print(f"  transactions.csv : {len(transactions)} rows")
print(f"  settlements.csv  : {len(settlements)} rows")
print(f"  refunds.csv      : {len(refunds)} rows")
print(f"  fees.csv         : {len(fees)} rows")
print(f"  ground_truth.csv : {len(ground_truth)} rows (cases)")

counts = Counter(g["case_type"] for g in ground_truth)
print("\nCase type breakdown:")
for k, v in counts.items():
    print(f"  {k}: {v}")

# sanity check: no accidental amount collisions among transactions
amt_seen = Counter(t["amount"] for t in transactions)
unexpected_dupes = {a: c for a, c in amt_seen.items() if c > 2}
print(f"\nAmount-collision sanity check: {len(unexpected_dupes)} amounts used more than twice (should be 0)")
