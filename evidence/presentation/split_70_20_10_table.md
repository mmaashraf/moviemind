# Time-Based Split Table (Requested 70/20/10)

Requested ratios were 70/20/10.  
For leakage-safe chronology in tuning, pipeline uses:
- Train: oldest 70%
- Validation: next 10%
- Test: most recent 20%

Equivalent ratio statement:
- Train = 70%
- Test = 20%
- Validation = 10%

| Split | Target Ratio | Rows | Actual Ratio | Min Timestamp | Max Timestamp |
|---|---:|---:|---:|---:|---:|
| train | 0.70 | 700146 | 0.700000 | 956703932 | 974862386 |
| val | 0.10 | 100020 | 0.099999 | 974862386 | 975768738 |
| test | 0.20 | 200043 | 0.200001 | 975768738 | 1046454590 |

Total rows: 1000209

Note: This split keeps test untouched for final evaluation while validation is used for tuning.
