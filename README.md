# OPAL: Aggregation-aware Backdoor Signal Analysis in Federated Learning

OPAL is a compact research-engineering project built from a larger FCBA+CLPBA experiment workspace. It studies why clean-label physical backdoor signals often fail after local training, submitted-update scaling, FedAvg aggregation, and post-attack benign recovery.

The value of this project is the experimental architecture: it adds submitted-update construction, trigger-direction write-in, APD-style virtual aggregation diagnostics, retention-memory variants, subspace-restricted write-in configs, metric logging, evaluators, and focused tests.

## Highlights

- Built an end-to-end experiment path for clean-label physical backdoor analysis in federated learning.
- Identified the key failure boundary: local poison success does not guarantee aggregation-facing submitted updates preserve the trigger direction.
- Implemented submitted-update write-in, which produced strong global ASR during attack rounds.
- Evaluated retention variants and showed benign recovery can overwrite the backdoor, motivating subspace-restricted write-in.
- Added formal CSV/JSON evaluators and unit tests for finite metrics, norm preservation, config contracts, and write-in behavior.

## Selected results

All rows use seed1/IID, `attack_stop_epoch=20`, `epochs=40`, SCD disabled, the same malicious-client setting, and the same submitted-update L2 cap.

| Stage | Method variable | ASR@stop | ASR-t20 | Clean@40 | Takeaway |
|---|---:|---:|---:|---:|---|
| Stage5o | submitted-update write-in, alpha=0.05 | 100.0 | 28.6 | 59.5 | Global write-in works; retention fails |
| Stage5p | benign-momentum compensation, gamma=0.25 | 100.0 | 26.4 | 57.01 | Coarse recovery compensation is not enough |
| Stage5q | submitted-update memory, weight=0.25 | 100.0 | 22.4 | 58.98 | Cross-round update memory is not enough |

The next branch, Stage5r, restricts submitted-update write-in to `layer4 + linear` parameters to test whether a more persistent subspace improves retention.

## Repository layout

```text
src/opal/                  Core OPAL utilities
experiments/configs/       Selected Stage5o/5p/5q/5r configs
experiments/evaluators/    Report generator for submitted-write-in evidence
experiments/runners/       Sanitized protocol runner
tests/                     Lightweight verification tests
docs/                      Concise method and result notes
results/                   Small sanitized result summaries
```

## Quick verification

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests exercise the showcase utilities directly and do not require datasets, checkpoints, or remote GPU access.

## Acknowledgements

This project was built from a larger research workspace integrating federated backdoor learning and clean-label physical backdoor attack experiments. See `docs/references.md` for the compact note.

## Notes

This repository is a curated showcase, not the full private experiment archive. Large datasets, model checkpoints, raw remote logs, and internal handoff notes are intentionally excluded.
