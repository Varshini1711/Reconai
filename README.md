# ReconAI

AI-powered financial reconciliation system for matching payment transactions with settlements, refunds, and fees while identifying cases that require human review.

## Problem

In digital payments, financial data is distributed across multiple systems such as transactions, settlements, refunds, and fees.

These records do not always map directly to each other. A single transaction may be settled after a delay, multiple transactions may be grouped into one settlement, refunds may reduce the final settlement amount, and fees may cause the settlement amount to differ from the original transaction.

A reconciliation system needs to determine which records belong together without creating false financial matches.

The key challenge is not only matching records, but also knowing when the available evidence is insufficient and escalating those cases instead of guessing.

## Solution

ReconAI is a hybrid reconciliation system that combines deterministic matching with an agentic AI investigator.

The system first resolves straightforward cases using rule-based reconciliation. Cases that remain ambiguous are passed to a Gemini-powered agent that investigates the available records using tools, verifies financial relationships, performs self-critique, and produces a final decision.

Every record is classified as:

- `AUTO_MATCHED` — sufficient evidence exists for an automatic match
- `NEEDS_REVIEW` — the system identifies a possible relationship but requires human verification
- `UNRESOLVED` — there is not enough evidence to establish a reliable match

ReconAI is designed to prefer an honest exception over a confident but incorrect financial match.

## Architecture

![ReconAI Architecture](architecture.png)

The reconciliation pipeline consists of five main layers:

### Layer 1 — Exact Matching

Handles straightforward cases using deterministic rules such as:

- Exact transaction and settlement identifiers
- Merchant matching
- Amount matching
- Date validation

### Layer 2 — Candidate Generation

Searches for potential settlement records based on relevant transaction attributes such as merchant and amount.

This reduces the search space before more expensive matching techniques are applied.

### Layer 3 — Fuzzy Matching

Handles small variations in merchant names and other noisy fields using fuzzy similarity.

For example, variations in merchant naming can still be considered when the financial attributes remain consistent.

### Layer 4 — Grouped Reconciliation

Handles financial relationships that are not one-to-one, including:

- Many transactions → one settlement
- One transaction → multiple settlements
- Partial settlements
- Refund-adjusted settlements
- Fee-adjusted settlements

This layer accounts for the fact that payment systems frequently batch or split financial records.

### Layer 5 — Agentic Investigation

Cases that remain ambiguous are investigated by a Gemini-powered agent.

The agent can:

1. Inspect transaction records
2. Search candidate records
3. Inspect settlements
4. Inspect refunds and fees
5. Verify financial relationships
6. Check date differences
7. Self-critique the proposed decision
8. Produce a final decision or escalate the case

The agent does not simply generate a response. It chooses investigation tools based on the case and uses the resulting evidence before making a decision.

## Agent Safety

ReconAI does not allow the LLM to blindly override reconciliation safety rules.

After the Gemini agent proposes an automatic match, additional validation checks are applied before the result is accepted.

The system verifies factors such as:

- Proposed settlement availability
- Merchant similarity
- Transaction and settlement date ordering

If a proposed match fails these checks, it is escalated to human review.

This creates a separation between AI investigation and financial decision safety.

## Evaluation

ReconAI was evaluated on a development dataset containing 71 cases.

| Metric | Result |
|---|---:|
| Auto-match precision | 94.2% |
| Coverage | 69.0% |
| Escalation correctness | 62.5% |
| Unresolved accuracy | 90.9% |
| False match rate | 5.8% |

The evaluation includes different reconciliation scenarios such as:

- Clean one-to-one matches
- Merchant name noise
- Ambiguous same-amount transactions
- Batched settlements
- Duplicate records
- Missing settlements
- One-to-many relationships
- Many-to-one relationships
- Partial settlements
- Refund-adjusted settlements
- Fee-adjusted settlements

The goal is not to maximize the number of automatic matches at any cost. ReconAI prioritizes reliable matches and explicit exception handling.

## Example

A transaction may contain:

```text
Transaction amount: ₹9,710
Merchant: BookMyShow
Transaction date: 2026-09-02
