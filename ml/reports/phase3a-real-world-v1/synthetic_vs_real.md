# Synthetic vs real-world transfer

Synthetic = Phase 2 held-out synthetic test split. Real = VizWiz val subset, primary vote threshold 3. Macro-F1 comparison restricted to the 4 evaluable labels.

| label | syn F1 | real F1 | Δ | syn P/R | real P/R | real support |
|---|---|---|---|---|---|---|
| blur | 0.9004 | 0.6108 | 0.29 | 0.9811/0.832 | 0.5077/0.7664 | 732 |
| overexposure | 0.766 | 0.0978 | 0.668 | 0.7912/0.7423 | 0.0583/0.303 | 66 |
| underexposure | 0.8485 | 0.4776 | 0.371 | 0.9333/0.7778 | 0.4923/0.4638 | 69 |
| defect | 0.4596 | 0.03 | 0.43 | 0.4154/0.5143 | 0.0156/0.3864 | 44 |

Macro-F1 (4 evaluable labels): synthetic **0.7436** -> real **0.304**.
