import csv
import copy
import math

train_fileHeader = ["local_model", "round", "epoch", "internal_epoch", "average_loss", "accuracy", "correct_data",
                    "total_data"]
test_fileHeader = ["model", "epoch", "average_loss", "accuracy", "correct_data", "total_data"]
train_result = []  # train_fileHeader
test_result = []  # test_fileHeader
posiontest_result = []  # test_fileHeader

triggertest_fileHeader = ["model", "trigger_name", "trigger_value", "epoch", "average_loss", "accuracy", "correct_data",
                          "total_data"]
poisontriggertest_result = []  # triggertest_fileHeader

posion_test_result = []  # train_fileHeader
posion_posiontest_result = []  # train_fileHeader
weight_result=[]
scale_result=[]
scale_temp_one_row=[]

experiment_metadata_fileHeader = ["key", "value"]
clean_label_metrics_fileHeader = [
    "round",
    "agent",
    "mode",
    "poison_count",
    "source_class",
    "target_class",
    "changed_labels",
    "changed_pixels",
]
ogm_metrics_fileHeader = [
    "round",
    "ogm_active",
    "cosine",
    "penalty",
    "lambda_orth",
    "momentum_available",
    "momentum_norm",
]
asr_t_fileHeader = ["round", "round_offset", "clean_acc", "asr", "attack_active"]
scd_metrics_fileHeader = [
    "round",
    "agent",
    "group",
    "mask_type",
    "active_parameters",
    "total_parameters",
    "apply_scope",
    "shared_parameters",
    "private_parameters",
    "shared_fraction",
    "active_fraction",
]
scd_update_metrics_fileHeader = [
    "round",
    "agent",
    "mode",
    "input_norm",
    "projected_norm",
    "compensated_norm",
    "final_norm",
    "norm_bound",
    "clip_ratio",
    "active_fraction",
    "min_coverage",
    "max_coverage",
    "finite",
]
update_similarity_metrics_fileHeader = [
    "round",
    "first_agent",
    "second_agent",
    "cosine",
]
aggregation_alignment_metrics_fileHeader = [
    "round",
    "malicious_client_count",
    "benign_client_count",
    "malicious_aggregate_norm",
    "benign_aggregate_norm",
    "total_aggregate_norm",
    "applied_total_update_norm",
    "malicious_to_benign_cosine",
    "malicious_to_total_cosine",
    "benign_to_total_cosine",
    "eta_over_no_models",
    "finite",
]
signal_trace_metrics_fileHeader = [
    "round",
    "agent",
    "mode",
    "boundary",
    "gm_loss",
    "poison_trigger_cosine",
    "poison_grad_norm",
    "trigger_grad_norm",
    "update_norm",
    "ogm_cosine",
    "ogm_penalty",
    "finite",
]
submission_stability_metrics_fileHeader = [
    "round",
    "agent",
    "mode",
    "scale",
    "scale_model_buffers",
    "submitted_update_l2_cap",
    "pre_scale_model_norm",
    "post_scale_model_norm",
    "parameter_update_norm",
    "buffer_update_norm",
    "finite_state",
    "post_scale_clean_acc",
    "post_scale_local_asr",
]
submission_writein_metrics_fileHeader = [
    "round",
    "agent",
    "alpha",
    "applied",
    "pre_update_norm",
    "post_update_norm",
    "source_grad_norm",
    "pre_descent_cosine",
    "post_descent_cosine",
    "pre_descent_projection_ratio",
    "post_descent_projection_ratio",
    "finite",
    "retention_gamma",
    "retention_vector_norm",
    "retention_compensation_applied",
    "target_vector_norm",
    "pre_target_cosine",
    "post_target_cosine",
    "post_minus_pre_target_cosine",
    "retention_memory_weight",
    "retention_memory_norm",
    "retention_memory_applied",
    "writein_scope",
    "writein_selected_parameter_count",
    "writein_total_parameter_count",
    "writein_selected_fraction",
]
apd_metrics_fileHeader = [
    "round",
    "agent",
    "mode",
    "virtual_backdoor_loss",
    "poison_surrogate_loss",
    "temporal_cosine",
    "benign_drift_norm",
    "raw_malicious_update_norm",
    "malicious_update_norm",
    "memory_delta_norm",
    "eta_over_no_models",
    "local_lr_proxy",
    "trigger_descent_projection_reward",
    "trigger_projection_lambda",
    "trigger_descent_projection_loss_component",
    "finite",
]
trigger_projection_metrics_fileHeader = [
    "round",
    "scope",
    "projection_reference",
    "malicious_projection",
    "benign_projection",
    "total_projection",
    "malicious_projection_ratio",
    "benign_projection_ratio",
    "total_projection_ratio",
    "malicious_cosine",
    "benign_cosine",
    "total_cosine",
    "finite",
]
virtual_actual_gap_metrics_fileHeader = [
    "round",
    "agent",
    "virtual_backdoor_loss",
    "actual_post_aggregation_backdoor_loss",
    "virtual_actual_loss_gap",
    "virtual_asr_proxy",
    "actual_global_asr",
    "finite",
]

experiment_metadata = []
clean_label_metrics = []
ogm_metrics = []
asr_t_result = []
scd_metrics = []
scd_update_metrics = []
update_similarity_metrics = []
aggregation_alignment_metrics = []
signal_trace_metrics = []
submission_stability_metrics = []
submission_writein_metrics = []
apd_metrics = []
trigger_projection_metrics = []
virtual_actual_gap_metrics = []


def reset_experiment_metrics():
    experiment_metadata.clear()
    clean_label_metrics.clear()
    ogm_metrics.clear()
    asr_t_result.clear()
    scd_metrics.clear()
    scd_update_metrics.clear()
    update_similarity_metrics.clear()
    aggregation_alignment_metrics.clear()
    signal_trace_metrics.clear()
    submission_stability_metrics.clear()
    submission_writein_metrics.clear()
    apd_metrics.clear()
    trigger_projection_metrics.clear()
    virtual_actual_gap_metrics.clear()


def record_experiment_metadata(params_file, params, git_commit):
    experiment_metadata.clear()
    seed = params.get('seed', params.get('manual_seed', ''))
    metadata = [
        ["git_commit", str(git_commit)],
        ["params_file", str(params_file)],
        ["seed", str(seed)],
        ["environment_name", str(params.get('environment_name', ''))],
        ["experiment_id", str(params.get('experiment_id', params.get('environment_name', '')))],
        ["experiment_family", str(params.get('experiment_family', ''))],
    ]
    experiment_metadata.extend(metadata)


def record_clean_label_metric(round_idx, agent_name_key, mode, poison_count, source_class, target_class,
                              changed_labels, changed_pixels):
    clean_label_metrics.append([
        round_idx,
        agent_name_key,
        mode,
        poison_count,
        source_class,
        target_class,
        changed_labels,
        changed_pixels,
    ])


def record_ogm_metric(round_idx, ogm_stats, momentum_stats=None):
    if ogm_stats is None:
        return
    momentum_stats = momentum_stats or {}
    ogm_metrics.append([
        round_idx,
        ogm_stats.get('ogm_active', False),
        ogm_stats.get('poison_momentum_cosine', ''),
        ogm_stats.get('orthogonal_penalty', ''),
        ogm_stats.get('lambda_orth', ''),
        momentum_stats.get('momentum_available', ''),
        momentum_stats.get('momentum_norm', ''),
    ])


def should_record_asr_t(params, round_idx):
    attack_stop_epoch = params.get('attack_stop_epoch')
    if attack_stop_epoch is None:
        return False
    round_offset = int(round_idx) - int(attack_stop_epoch)
    return round_offset in [int(offset) for offset in params.get('asr_t_checkpoints', [])]


def record_asr_t_result(round_idx, attack_stop_epoch, clean_acc, asr, attack_active):
    asr_t_result.append([
        round_idx,
        int(round_idx) - int(attack_stop_epoch),
        clean_acc,
        asr,
        attack_active,
    ])


def record_scd_metric(round_idx, agent_name_key, group_id, mask_type, active_parameters, total_parameters,
                      apply_scope="attack_only", shared_parameters=0, private_parameters=None,
                      shared_fraction=0.0, active_fraction=None):
    if private_parameters is None:
        private_parameters = active_parameters - shared_parameters
    if active_fraction is None:
        active_fraction = active_parameters / total_parameters
    scd_metrics.append([
        round_idx,
        agent_name_key,
        group_id,
        mask_type,
        active_parameters,
        total_parameters,
        apply_scope,
        shared_parameters,
        private_parameters,
        shared_fraction,
        active_fraction,
    ])


def record_scd_update_metric(round_idx, agent_name_key, mode, stats):
    numeric_fields = (
        "input_norm",
        "projected_norm",
        "compensated_norm",
        "final_norm",
        "norm_bound",
        "clip_ratio",
        "active_fraction",
    )
    finite = all(math.isfinite(float(stats[field])) for field in numeric_fields)
    scd_update_metrics.append([
        round_idx,
        agent_name_key,
        mode,
        stats["input_norm"],
        stats["projected_norm"],
        stats["compensated_norm"],
        stats["final_norm"],
        stats["norm_bound"],
        stats["clip_ratio"],
        stats["active_fraction"],
        stats.get("min_coverage", 1),
        stats.get("max_coverage", 1),
        finite,
    ])


def record_update_similarity(round_idx, first_agent, second_agent, cosine):
    if not math.isfinite(float(cosine)):
        raise ValueError("Update cosine must be finite.")
    update_similarity_metrics.append([
        round_idx,
        first_agent,
        second_agent,
        cosine,
    ])


def record_aggregation_alignment_metric(round_idx, stats):
    aggregation_alignment_metrics.append([
        round_idx,
        stats.get("malicious_client_count", ""),
        stats.get("benign_client_count", ""),
        stats.get("malicious_aggregate_norm", ""),
        stats.get("benign_aggregate_norm", ""),
        stats.get("total_aggregate_norm", ""),
        stats.get("applied_total_update_norm", ""),
        stats.get("malicious_to_benign_cosine", ""),
        stats.get("malicious_to_total_cosine", ""),
        stats.get("benign_to_total_cosine", ""),
        stats.get("eta_over_no_models", ""),
        bool(stats.get("finite", True)),
    ])


def record_signal_trace_metric(round_idx, agent_name_key, mode, boundary, metrics):
    finite = bool(metrics.get("finite", True))
    signal_trace_metrics.append([
        round_idx,
        agent_name_key,
        mode,
        boundary,
        metrics.get("gm_loss", ""),
        metrics.get("poison_trigger_cosine", ""),
        metrics.get("poison_grad_norm", ""),
        metrics.get("trigger_grad_norm", ""),
        metrics.get("update_norm", ""),
        metrics.get("ogm_cosine", ""),
        metrics.get("ogm_penalty", ""),
        finite,
    ])


def record_submission_stability_metric(round_idx, agent_name_key, stats):
    submission_stability_metrics.append([
        round_idx,
        agent_name_key,
        stats.get("mode", ""),
        stats.get("scale", ""),
        stats.get("scale_model_buffers", ""),
        stats.get("submitted_update_l2_cap", ""),
        stats.get("pre_scale_model_norm", ""),
        stats.get("post_scale_model_norm", ""),
        stats.get("parameter_update_norm", ""),
        stats.get("buffer_update_norm", ""),
        stats.get("finite_state", ""),
        stats.get("post_scale_clean_acc", ""),
        stats.get("post_scale_local_asr", ""),
    ])


def record_submission_writein_metric(round_idx, agent_name_key, stats):
    submission_writein_metrics.append([
        round_idx,
        agent_name_key,
        stats.get("alpha", ""),
        bool(stats.get("applied", False)),
        stats.get("pre_update_norm", ""),
        stats.get("post_update_norm", ""),
        stats.get("source_grad_norm", ""),
        stats.get("pre_descent_cosine", ""),
        stats.get("post_descent_cosine", ""),
        stats.get("pre_descent_projection_ratio", ""),
        stats.get("post_descent_projection_ratio", ""),
        bool(stats.get("finite", True)),
        stats.get("retention_gamma", ""),
        stats.get("retention_vector_norm", ""),
        bool(stats.get("retention_compensation_applied", False)),
        stats.get("target_vector_norm", ""),
        stats.get("pre_target_cosine", ""),
        stats.get("post_target_cosine", ""),
        stats.get("post_minus_pre_target_cosine", ""),
        stats.get("retention_memory_weight", ""),
        stats.get("retention_memory_norm", ""),
        bool(stats.get("retention_memory_applied", False)),
        stats.get("writein_scope", "full"),
        stats.get("writein_selected_parameter_count", ""),
        stats.get("writein_total_parameter_count", ""),
        stats.get("writein_selected_fraction", ""),
    ])


def record_apd_metric(round_idx, agent_name_key, mode, stats):
    apd_metrics.append([
        round_idx,
        agent_name_key,
        mode,
        stats.get("virtual_backdoor_loss", ""),
        stats.get("poison_surrogate_loss", ""),
        stats.get("temporal_cosine", ""),
        stats.get("benign_drift_norm", ""),
        stats.get("raw_malicious_update_norm", ""),
        stats.get("malicious_update_norm", ""),
        stats.get("memory_delta_norm", ""),
        stats.get("eta_over_no_models", ""),
        stats.get("local_lr_proxy", ""),
        stats.get("trigger_descent_projection_reward", ""),
        stats.get("trigger_projection_lambda", ""),
        stats.get("trigger_descent_projection_loss_component", ""),
        bool(stats.get("finite", True)),
    ])


def record_trigger_projection_metric(round_idx, stats):
    trigger_projection_metrics.append([
        round_idx,
        stats.get("scope", ""),
        stats.get("projection_reference", ""),
        stats.get("malicious_projection", ""),
        stats.get("benign_projection", ""),
        stats.get("total_projection", ""),
        stats.get("malicious_projection_ratio", ""),
        stats.get("benign_projection_ratio", ""),
        stats.get("total_projection_ratio", ""),
        stats.get("malicious_cosine", ""),
        stats.get("benign_cosine", ""),
        stats.get("total_cosine", ""),
        bool(stats.get("finite", True)),
    ])


def record_virtual_actual_gap_metric(round_idx, agent_name_key, stats):
    virtual_actual_gap_metrics.append([
        round_idx,
        agent_name_key,
        stats.get("virtual_backdoor_loss", ""),
        stats.get("actual_post_aggregation_backdoor_loss", ""),
        stats.get("virtual_actual_loss_gap", ""),
        stats.get("virtual_asr_proxy", ""),
        stats.get("actual_global_asr", ""),
        bool(stats.get("finite", True)),
    ])


def _write_csv(folder_path, filename, header, rows):
    csv_file = open(f'{folder_path}/{filename}', "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(header)
    writer.writerows(rows)
    csv_file.close()


def save_result_csv(epoch, is_posion,folder_path):
    _write_csv(folder_path, 'train_result.csv', train_fileHeader, train_result)

    _write_csv(folder_path, 'test_result.csv', test_fileHeader, test_result)

    if len(experiment_metadata) > 0:
        _write_csv(folder_path, 'experiment_metadata.csv', experiment_metadata_fileHeader, experiment_metadata)

    if len(clean_label_metrics) > 0:
        _write_csv(folder_path, 'clean_label_metrics.csv', clean_label_metrics_fileHeader, clean_label_metrics)

    if len(ogm_metrics) > 0:
        _write_csv(folder_path, 'ogm_metrics.csv', ogm_metrics_fileHeader, ogm_metrics)

    if len(asr_t_result) > 0:
        _write_csv(folder_path, 'asr_t_result.csv', asr_t_fileHeader, asr_t_result)

    if len(scd_metrics) > 0:
        _write_csv(folder_path, 'scd_metrics.csv', scd_metrics_fileHeader, scd_metrics)

    if len(scd_update_metrics) > 0:
        _write_csv(
            folder_path,
            'scd_update_metrics.csv',
            scd_update_metrics_fileHeader,
            scd_update_metrics,
        )

    if len(update_similarity_metrics) > 0:
        _write_csv(
            folder_path,
            'update_similarity_metrics.csv',
            update_similarity_metrics_fileHeader,
            update_similarity_metrics,
        )

    if len(aggregation_alignment_metrics) > 0:
        _write_csv(
            folder_path,
            'aggregation_alignment_metrics.csv',
            aggregation_alignment_metrics_fileHeader,
            aggregation_alignment_metrics,
        )

    if len(signal_trace_metrics) > 0:
        _write_csv(
            folder_path,
            'signal_trace_metrics.csv',
            signal_trace_metrics_fileHeader,
            signal_trace_metrics,
        )

    if len(submission_stability_metrics) > 0:
        _write_csv(
            folder_path,
            'submission_stability_metrics.csv',
            submission_stability_metrics_fileHeader,
            submission_stability_metrics,
        )

    if len(submission_writein_metrics) > 0:
        _write_csv(
            folder_path,
            'submission_writein_metrics.csv',
            submission_writein_metrics_fileHeader,
            submission_writein_metrics,
        )

    if len(apd_metrics) > 0:
        _write_csv(
            folder_path,
            'apd_metrics.csv',
            apd_metrics_fileHeader,
            apd_metrics,
        )

    if len(trigger_projection_metrics) > 0:
        _write_csv(
            folder_path,
            'trigger_projection_metrics.csv',
            trigger_projection_metrics_fileHeader,
            trigger_projection_metrics,
        )

    if len(virtual_actual_gap_metrics) > 0:
        _write_csv(
            folder_path,
            'virtual_actual_gap_metrics.csv',
            virtual_actual_gap_metrics_fileHeader,
            virtual_actual_gap_metrics,
        )

    if len(weight_result)>0:
        weight_csvFile=  open(f'{folder_path}/weight_result.csv', "w", newline="")
        weight_writer = csv.writer(weight_csvFile)
        weight_writer.writerows(weight_result)
        weight_csvFile.close()

    if len(scale_temp_one_row)>0:
        _csvFile=  open(f'{folder_path}/scale_result.csv', "w", newline="")
        _writer = csv.writer(_csvFile)
        scale_result.append(copy.deepcopy(scale_temp_one_row))
        scale_temp_one_row.clear()
        _writer.writerows(scale_result)
        _csvFile.close()

    if is_posion:
        _write_csv(folder_path, 'posiontest_result.csv', test_fileHeader, posiontest_result)
        _write_csv(folder_path, 'poisontriggertest_result.csv', triggertest_fileHeader, poisontriggertest_result)

def add_weight_result(name,weight,alpha):
    weight_result.append(name)
    weight_result.append(weight)
    weight_result.append(alpha)
