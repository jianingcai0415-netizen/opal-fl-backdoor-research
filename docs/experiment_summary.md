# Experiment summary

The main experimental result is not that every method succeeded. The important result is that the project isolated two different bottlenecks:

- local-to-global transfer: whether an attack signal appears in the actual submitted update and survives FedAvg;
- benign-recovery retention: whether the backdoor remains after malicious updates stop.

## Retention gates

| Stage | Isolated variable | ASR@stop | ASR-t20 | Clean@40 | Interpretation |
|---|---|---:|---:|---:|---|
| Stage5o | submitted-update write-in | 100.0 | 28.6 | 59.5 | Solves local-to-global transfer but not retention |
| Stage5p | benign-momentum compensation | 100.0 | 26.4 | 57.01 | Compensation vector was too coarse |
| Stage5q | submitted-update memory | 100.0 | 22.4 | 58.98 | Memory did not prevent benign overwrite |

These gates justify Stage5r: keep the submitted-update boundary fixed, but restrict the write-in to a potentially more persistent parameter subspace.

