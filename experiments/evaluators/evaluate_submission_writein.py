import argparse
import csv
import json
import math
import statistics
from pathlib import Path


REQUIRED_FILES = [
    "test_result.csv",
    "posiontest_result.csv",
    "submission_stability_metrics.csv",
    "aggregation_alignment_metrics.csv",
    "trigger_projection_metrics.csv",
    "submission_writein_metrics.csv",
    "apd_metrics.csv",
]


class MissingStage5oEvidenceError(FileNotFoundError):
    pass


def _read_csv(path, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            raise MissingStage5oEvidenceError(path)
        return []
    with open(path, newline="", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        return list(csv.DictReader(f, delimiter=delimiter))


def _float_or_none(value):
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bool_value(value):
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def _max(values):
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _latest(values_by_epoch):
    if not values_by_epoch:
        return None
    return values_by_epoch[max(values_by_epoch)]


def _find_result_dir(run_dir):
    run_dir = Path(run_dir)
    if all((run_dir / filename).exists() for filename in REQUIRED_FILES[:2]):
        return run_dir

    model_dir = run_dir / "model_dir"
    if all((model_dir / filename).exists() for filename in REQUIRED_FILES[:2]):
        return model_dir

    saved_root = run_dir / "saved_models"
    if saved_root.exists():
        candidates = [
            path
            for path in sorted(saved_root.iterdir())
            if path.is_dir() and all((path / filename).exists() for filename in REQUIRED_FILES[:2])
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(f"Expected one Stage5o result directory, found {len(candidates)} in {saved_root}")

    raise MissingStage5oEvidenceError(run_dir / "model_dir")


def _find_log_root(run_dir):
    run_dir = Path(run_dir)
    log_root = run_dir / "log_root"
    return log_root if log_root.exists() else None


def _read_params(result_dir):
    params = {"attack_stop_epoch": 60, "asr_t_checkpoints": [20, 40]}
    path = Path(result_dir) / "params.yaml"
    if not path.exists():
        return params
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    line_idx = 0
    while line_idx < len(lines):
        raw_line = lines[line_idx]
        line_idx += 1
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "attack_stop_epoch":
            parsed = _float_or_none(value)
            if parsed is not None:
                params[key] = int(parsed)
        elif key == "asr_t_checkpoints" and value.startswith("[") and value.endswith("]"):
            params[key] = [
                int(float(item.strip()))
                for item in value[1:-1].split(",")
                if item.strip()
            ]
        elif key == "asr_t_checkpoints" and value == "":
            checkpoints = []
            while line_idx < len(lines):
                item_line = lines[line_idx]
                stripped = item_line.strip()
                if not stripped:
                    line_idx += 1
                    continue
                if not stripped.startswith("-"):
                    break
                parsed = _float_or_none(stripped[1:].strip())
                if parsed is not None:
                    checkpoints.append(int(parsed))
                line_idx += 1
            params[key] = checkpoints
    return params


def _global_accuracy_by_epoch(result_dir, filename):
    by_epoch = {}
    for row in _read_csv(Path(result_dir) / filename):
        if row.get("model") != "global":
            continue
        epoch = _float_or_none(row.get("epoch"))
        accuracy = _float_or_none(row.get("accuracy"))
        if epoch is not None and accuracy is not None:
            by_epoch[int(epoch)] = accuracy
    return dict(sorted(by_epoch.items()))


def _count_nonfinite(rows, finite_column):
    return sum(1 for row in rows if not _bool_value(row.get(finite_column, True)))


def _first_epoch_at_or_above(values_by_epoch, threshold):
    for epoch, value in values_by_epoch.items():
        if value >= threshold:
            return epoch
    return None


def _summarize_global(result_dir):
    clean_by_epoch = _global_accuracy_by_epoch(result_dir, "test_result.csv")
    asr_by_epoch = _global_accuracy_by_epoch(result_dir, "posiontest_result.csv")
    return {
        "global_clean_by_epoch": clean_by_epoch,
        "global_asr_by_epoch": asr_by_epoch,
        "global_clean_latest": _latest(clean_by_epoch),
        "global_asr_latest": _latest(asr_by_epoch),
        "global_clean_final": _latest(clean_by_epoch),
        "global_asr_max": _max(asr_by_epoch.values()),
        "first_asr_ge_50_epoch": _first_epoch_at_or_above(asr_by_epoch, 50.0),
        "first_asr_ge_95_epoch": _first_epoch_at_or_above(asr_by_epoch, 95.0),
        "completed_epochs": max(clean_by_epoch) if clean_by_epoch else 0,
    }


def _summarize_writein(result_dir):
    rows = _read_csv(Path(result_dir) / "submission_writein_metrics.csv")
    pre_cosines = [_float_or_none(row.get("pre_descent_cosine")) for row in rows]
    post_cosines = [_float_or_none(row.get("post_descent_cosine")) for row in rows]
    pre_ratios = [_float_or_none(row.get("pre_descent_projection_ratio")) for row in rows]
    post_ratios = [_float_or_none(row.get("post_descent_projection_ratio")) for row in rows]
    retention_norms = [_float_or_none(row.get("retention_vector_norm")) for row in rows]
    retention_memory_norms = [_float_or_none(row.get("retention_memory_norm")) for row in rows]
    selected_fractions = [_float_or_none(row.get("writein_selected_fraction")) for row in rows]
    pre_target_cosines = [_float_or_none(row.get("pre_target_cosine")) for row in rows]
    post_target_cosines = [_float_or_none(row.get("post_target_cosine")) for row in rows]
    target_cosine_deltas = [_float_or_none(row.get("post_minus_pre_target_cosine")) for row in rows]
    norm_drifts = []
    alphas = set()
    retention_gammas = set()
    retention_memory_weights = set()
    writein_scopes = set()
    applied_count = 0
    retention_applied_count = 0
    retention_memory_applied_count = 0
    for row in rows:
        writein_scopes.add(row.get("writein_scope") or "full")
        alpha = _float_or_none(row.get("alpha"))
        if alpha is not None:
            alphas.add(alpha)
        retention_gamma = _float_or_none(row.get("retention_gamma"))
        if retention_gamma is not None:
            retention_gammas.add(retention_gamma)
        retention_memory_weight = _float_or_none(row.get("retention_memory_weight"))
        if retention_memory_weight is not None:
            retention_memory_weights.add(retention_memory_weight)
        if _bool_value(row.get("applied", False)):
            applied_count += 1
        if _bool_value(row.get("retention_compensation_applied", False)):
            retention_applied_count += 1
        if _bool_value(row.get("retention_memory_applied", False)):
            retention_memory_applied_count += 1
        pre_norm = _float_or_none(row.get("pre_update_norm"))
        post_norm = _float_or_none(row.get("post_update_norm"))
        if pre_norm is not None and post_norm is not None:
            norm_drifts.append(abs(post_norm - pre_norm))
    return {
        "row_count": len(rows),
        "alpha_values": sorted(alphas),
        "applied_count": applied_count,
        "pre_descent_cosine_mean": _mean(pre_cosines),
        "post_descent_cosine_mean": _mean(post_cosines),
        "post_minus_pre_descent_cosine_mean": (
            _mean(post_cosines) - _mean(pre_cosines)
            if _mean(post_cosines) is not None and _mean(pre_cosines) is not None
            else None
        ),
        "pre_descent_projection_ratio_mean": _mean(pre_ratios),
        "post_descent_projection_ratio_mean": _mean(post_ratios),
        "norm_drift_max": _max(norm_drifts),
        "retention_gamma_values": sorted(retention_gammas),
        "retention_compensation_applied_count": retention_applied_count,
        "retention_vector_norm_latest": (
            [value for value in retention_norms if value is not None][-1]
            if any(value is not None for value in retention_norms)
            else None
        ),
        "retention_memory_weight_values": sorted(retention_memory_weights),
        "retention_memory_applied_count": retention_memory_applied_count,
        "retention_memory_norm_latest": (
            [value for value in retention_memory_norms if value is not None][-1]
            if any(value is not None for value in retention_memory_norms)
            else None
        ),
        "writein_scope_values": sorted(writein_scopes),
        "writein_selected_fraction_latest": (
            [value for value in selected_fractions if value is not None][-1]
            if any(value is not None for value in selected_fractions)
            else 1.0
        ),
        "writein_selected_fraction_mean": (
            _mean(selected_fractions)
            if any(value is not None for value in selected_fractions)
            else 1.0
        ),
        "pre_target_cosine_mean": _mean(pre_target_cosines),
        "post_target_cosine_mean": _mean(post_target_cosines),
        "post_minus_pre_target_cosine_mean": _mean(target_cosine_deltas),
        "nonfinite_row_count": _count_nonfinite(rows, "finite"),
        "latest": rows[-1] if rows else None,
    }


def _summarize_local_submission(result_dir):
    rows = _read_csv(Path(result_dir) / "submission_stability_metrics.csv")
    local_asrs = [_float_or_none(row.get("post_scale_local_asr")) for row in rows]
    return {
        "row_count": len(rows),
        "post_scale_local_asr_mean": _mean(local_asrs),
        "post_scale_local_asr_max": _max(local_asrs),
        "post_scale_local_asr_latest": local_asrs[-1] if local_asrs else None,
        "nonfinite_row_count": _count_nonfinite(rows, "finite_state"),
        "latest": rows[-1] if rows else None,
    }


def _summarize_trigger_projection(result_dir):
    rows = _read_csv(Path(result_dir) / "trigger_projection_metrics.csv")
    total_cosines = [_float_or_none(row.get("total_cosine")) for row in rows]
    total_ratios = [_float_or_none(row.get("total_projection_ratio")) for row in rows]
    descent_cosines = [-value for value in total_cosines if value is not None]
    descent_ratios = [-value for value in total_ratios if value is not None]
    return {
        "row_count": len(rows),
        "total_descent_cosine_mean": _mean(descent_cosines),
        "total_descent_cosine_latest": descent_cosines[-1] if descent_cosines else None,
        "total_descent_projection_ratio_mean": _mean(descent_ratios),
        "total_descent_projection_ratio_latest": descent_ratios[-1] if descent_ratios else None,
        "nonfinite_row_count": _count_nonfinite(rows, "finite"),
        "latest": rows[-1] if rows else None,
    }


def _summarize_alignment(result_dir):
    rows = _read_csv(Path(result_dir) / "aggregation_alignment_metrics.csv")
    malicious_to_total = [_float_or_none(row.get("malicious_to_total_cosine")) for row in rows]
    return {
        "row_count": len(rows),
        "malicious_to_total_cosine_mean": _mean(malicious_to_total),
        "malicious_to_total_cosine_latest": malicious_to_total[-1] if malicious_to_total else None,
        "nonfinite_row_count": _count_nonfinite(rows, "finite"),
        "latest": rows[-1] if rows else None,
    }


def _summarize_apd(result_dir):
    rows = _read_csv(Path(result_dir) / "apd_metrics.csv")
    drift_norms = [_float_or_none(row.get("benign_drift_norm")) for row in rows]
    finite_drifts = [value for value in drift_norms if value is not None]
    return {
        "row_count": len(rows),
        "benign_drift_norm_latest": finite_drifts[-1] if finite_drifts else None,
        "benign_drift_norm_max": _max(finite_drifts),
        "benign_drift_nonzero_count": sum(1 for value in finite_drifts if abs(value) > 1e-12),
        "nonfinite_row_count": _count_nonfinite(rows, "finite"),
        "latest": rows[-1] if rows else None,
    }


def _train_exit(log_root):
    if log_root is None:
        return None
    path = log_root / "train.exit"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return int(text) if text.lstrip("-").isdigit() else text


def _classify(global_metrics, writein, local_submission, trigger_projection, evidence, params=None, asr_t=None):
    if evidence["missing_files"]:
        return "missing_evidence"
    if evidence["nonfinite_row_count"] > 0:
        return "nonfinite_evidence"
    if writein["applied_count"] == 0:
        return "submission_writein_not_applied"
    if (writein["post_minus_pre_descent_cosine_mean"] is not None) and (
        writein["post_minus_pre_descent_cosine_mean"] <= 0.0
    ):
        return "submission_writein_not_improving_descent"
    if global_metrics["global_asr_max"] is None:
        return "global_trace_missing"
    if global_metrics["global_asr_max"] < 50.0:
        if (local_submission["post_scale_local_asr_max"] or 0.0) >= 50.0:
            return "local_global_transfer_gap"
        return "attack_signal_too_weak"
    params = params or {}
    asr_t = asr_t or {}
    attack_stop_epoch = params.get("attack_stop_epoch")
    asr_stop = (
        global_metrics["global_asr_by_epoch"].get(attack_stop_epoch)
        if attack_stop_epoch is not None
        else None
    )
    if (
        asr_stop is not None
        and asr_stop >= 50.0
        and any(value is not None and value < 50.0 for value in asr_t.values())
    ):
        return "retention_failure_after_attack_stop"
    if (global_metrics["global_clean_latest"] is not None) and global_metrics["global_clean_latest"] < 70.0:
        return "attack_success_with_clean_learning_failure"
    if (trigger_projection["total_descent_cosine_mean"] is not None) and (
        trigger_projection["total_descent_cosine_mean"] <= 0.0
    ):
        return "aggregation_trigger_direction_misaligned"
    return "paper_candidate"


def evaluate_stage5o_submission_writein(run_dir, output_path=None):
    run_dir = Path(run_dir)
    try:
        result_dir = _find_result_dir(run_dir)
    except MissingStage5oEvidenceError:
        result_dir = run_dir / "model_dir"
    log_root = _find_log_root(run_dir)
    missing_files = [filename for filename in REQUIRED_FILES if not (result_dir / filename).exists()]

    if missing_files:
        result = {
            "passed": False,
            "mechanism_passed": False,
            "paper_ready_candidate": False,
            "limiting_factor": "missing_evidence",
            "run_dir": run_dir.as_posix(),
            "result_dir": result_dir.as_posix(),
            "log_root": log_root.as_posix() if log_root else None,
            "train_exit": _train_exit(log_root),
            "params": {},
            "global_metrics": {},
            "submission_writein": {},
            "local_submission": {},
            "trigger_projection": {},
            "aggregation_alignment": {},
            "apd": {},
            "evidence_checks": {
                "missing_files": missing_files,
                "nonfinite_row_count": 0,
            },
            "next_recommended_stage": "complete_stage5o_evidence_archive",
        }
    else:
        params = _read_params(result_dir)
        global_metrics = _summarize_global(result_dir)
        writein = _summarize_writein(result_dir)
        local_submission = _summarize_local_submission(result_dir)
        trigger_projection = _summarize_trigger_projection(result_dir)
        aggregation_alignment = _summarize_alignment(result_dir)
        apd = _summarize_apd(result_dir)
        nonfinite_row_count = sum(
            section["nonfinite_row_count"]
            for section in [writein, local_submission, trigger_projection, aggregation_alignment, apd]
        )
        evidence = {
            "missing_files": [],
            "nonfinite_row_count": nonfinite_row_count,
        }
        attack_stop_epoch = params["attack_stop_epoch"]
        asr_by_epoch = global_metrics["global_asr_by_epoch"]
        asr_t = {
            checkpoint: asr_by_epoch.get(attack_stop_epoch + checkpoint)
            for checkpoint in params["asr_t_checkpoints"]
        }
        limiting_factor = _classify(
            global_metrics,
            writein,
            local_submission,
            trigger_projection,
            evidence,
            params=params,
            asr_t=asr_t,
        )
        mechanism_passed = (
            nonfinite_row_count == 0
            and writein["applied_count"] > 0
            and (writein["post_minus_pre_descent_cosine_mean"] or 0.0) > 0.0
            and (global_metrics["global_asr_max"] or 0.0) >= 50.0
        )
        paper_ready_candidate = (
            mechanism_passed
            and (global_metrics["global_clean_latest"] or 0.0) >= 70.0
            and (asr_by_epoch.get(attack_stop_epoch) or 0.0) >= 50.0
            and all(value is not None and value >= 50.0 for value in asr_t.values())
        )
        result = {
            "passed": nonfinite_row_count == 0,
            "mechanism_passed": mechanism_passed,
            "paper_ready_candidate": paper_ready_candidate,
            "limiting_factor": limiting_factor,
            "run_dir": run_dir.as_posix(),
            "result_dir": result_dir.as_posix(),
            "log_root": log_root.as_posix() if log_root else None,
            "train_exit": _train_exit(log_root),
            "params": params,
            "asr_t": asr_t,
            "global_metrics": global_metrics,
            "submission_writein": writein,
            "local_submission": local_submission,
            "trigger_projection": trigger_projection,
            "aggregation_alignment": aggregation_alignment,
            "apd": apd,
            "evidence_checks": evidence,
            "next_recommended_stage": (
                "multi_seed_and_non_iid_expansion"
                if paper_ready_candidate
                else "restore_clean_accuracy_without_changing_poison_budget"
                if limiting_factor == "attack_success_with_clean_learning_failure"
                else "add_single_retention_variable_without_changing_poison_budget"
                if limiting_factor == "retention_failure_after_attack_stop"
                else "systematic_stage5o_limiting_factor_debug"
            ),
        }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Stage5o submission-aware write-in evidence.")
    parser.add_argument("--run-dir", required=True, help="Stage5o archive or result directory")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args(argv)

    result = evaluate_stage5o_submission_writein(args.run_dir, output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
