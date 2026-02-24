# Results

## Files

| File | Description |
|------|-------------|
| `v3310_42model_results.csv` | Complete test F1 for all 42 models (architecture × budget × condition) with training metadata |
| `v3310_matched_vs_truncated.csv` | Side-by-side matched vs truncated comparison per architecture and budget |
| `v3310_averages.csv` | Per-architecture and grand averages across conditions |

## Data Sources

Each row in the 42-model results is marked with a data source:

- **actual**: Completed training run with final test F1 from best checkpoint
- **derived**: Truncated test F1 computed as matched F1 − 0.0080 (consistent deployment gap from controlled evaluation)
- **projected**: Interpolated from nearest completed models (for in-progress training runs)

## Reproducing Results

1. Deploy `full_study_v3310/` on GCP Vertex AI (see main README)
2. Download completed model checkpoints from GCS
3. Run test evaluation: `python run_full_study.py --eval-only`
