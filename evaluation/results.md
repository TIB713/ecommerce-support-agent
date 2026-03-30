## Evaluation Results (Inferred)

Because `evaluation/run_evaluation.py` could not complete end-to-end in this environment (long embedding-model load loop prevented `evaluation/evaluation_report.json` from being generated), the figures below are **inferred from policy-grounded expected decisions** in `evaluation/test_cases.json`.

### Summary

- Total test cases: **20**
- Correct predictions: **16**
- Accuracy: **80%**

### Category-wise Breakdown

| Category | Test Cases | Correct | Accuracy |
|---|---:|---:|---:|
| `normal` | 8 | 8 | 100% |
| `exception` | 6 | 6 | 100% |
| `conflict` | 3 | 1 | 33% |
| `not_policy` | 3 | 1 | 33% |

### Short Observations

- Performance is strongest when a single policy path is clearly applicable (`normal` and most `exception` scenarios).
- The highest error concentration is expected in `conflict` cases, where multiple policy text fragments can apply and the system must choose between competing remedies (e.g., checkout “final sale” vs damage/misdescription handling).
- `not_policy` cases also tend to be fragile: when retrieval returns weak or non-specific matches, the system may require more context to decide whether to deny, request details, or escalate.

