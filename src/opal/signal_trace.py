import math

import torch


BUFFER_MARKERS = ("running_mean", "running_var", "num_batches_tracked")


def _flatten_tensors(tensors):
    flattened = []
    finite = True
    for tensor in tensors:
        if tensor is None:
            continue
        detached = tensor.detach()
        if detached.is_floating_point() or detached.is_complex():
            finite = finite and bool(torch.isfinite(detached).all().item())
        flattened.append(detached.double().contiguous().view(-1).cpu())
    if not flattened:
        return torch.empty(0), finite
    return torch.cat(flattened), finite


def tensor_l2_norm(tensor):
    if tensor is None:
        return 0.0
    vector, finite = _flatten_tensors([tensor])
    if not finite:
        return float("nan")
    return float(vector.double().pow(2).sum().sqrt().item())


def update_l2_norm(update_dict):
    tensors = []
    for name, tensor in update_dict.items():
        if any(marker in name for marker in BUFFER_MARKERS):
            continue
        if tensor.is_floating_point() or tensor.is_complex():
            tensors.append(tensor)
    vector, finite = _flatten_tensors(tensors)
    if not finite:
        return float("nan")
    return float(vector.double().pow(2).sum().sqrt().item())


def tensor_vector_cosine(first, second):
    first_vector, first_finite = _flatten_tensors([first])
    second_vector, second_finite = _flatten_tensors([second])
    if not first_finite or not second_finite:
        return float("nan")
    if first_vector.numel() != second_vector.numel():
        raise ValueError(
            f"Cosine inputs must have the same size: {first_vector.numel()} != {second_vector.numel()}"
        )
    first_norm = first_vector.norm()
    second_norm = second_vector.norm()
    if float(first_norm.item()) == 0.0 or float(second_norm.item()) == 0.0:
        return 0.0
    cosine = torch.nn.functional.cosine_similarity(first_vector, second_vector, dim=0)
    value = float(cosine.item())
    return value if math.isfinite(value) else float("nan")


def gradient_list_cosine(first_grad_list, second_grad_list):
    first_vector, first_finite = _flatten_tensors(first_grad_list)
    second_vector, second_finite = _flatten_tensors(second_grad_list)
    if not first_finite or not second_finite:
        return float("nan")
    return tensor_vector_cosine(first_vector, second_vector)


def gradient_list_l2_norm(grad_list):
    vector, finite = _flatten_tensors(grad_list)
    if not finite:
        return float("nan")
    return float(vector.double().pow(2).sum().sqrt().item())


def flatten_update_for_parameter_names(update_dict, parameter_names):
    values = []
    for name in parameter_names:
        if name not in update_dict:
            raise KeyError(f"Missing update tensor for parameter: {name}")
        values.append(update_dict[name].detach().double().contiguous().view(-1).cpu())
    if not values:
        return torch.empty(0)
    return torch.cat(values)


def parameter_update_names(update_dict):
    return [
        name
        for name, tensor in update_dict.items()
        if (tensor.is_floating_point() or tensor.is_complex())
        and not any(marker in name for marker in BUFFER_MARKERS)
    ]


def aggregation_alignment_metrics(updates_by_agent, malicious_agents, eta_over_no_models=1.0):
    """Summarize malicious-vs-benign geometry before FedAvg scaling."""
    agent_items = list(updates_by_agent.items())
    malicious_agents = set(malicious_agents)
    eta_over_no_models = float(eta_over_no_models)

    if not agent_items:
        return {
            "malicious_client_count": 0,
            "benign_client_count": 0,
            "malicious_aggregate_norm": 0.0,
            "benign_aggregate_norm": 0.0,
            "total_aggregate_norm": 0.0,
            "applied_total_update_norm": 0.0,
            "malicious_to_benign_cosine": 0.0,
            "malicious_to_total_cosine": 0.0,
            "benign_to_total_cosine": 0.0,
            "eta_over_no_models": eta_over_no_models,
            "finite": True,
        }

    names = parameter_update_names(agent_items[0][1])
    template = torch.zeros_like(flatten_update_for_parameter_names(agent_items[0][1], names))

    def sum_group(agent_predicate):
        vector = template.clone()
        count = 0
        for agent_name, update_dict in agent_items:
            if agent_predicate(agent_name):
                vector = vector + flatten_update_for_parameter_names(update_dict, names)
                count += 1
        return vector, count

    malicious_vector, malicious_count = sum_group(lambda agent_name: agent_name in malicious_agents)
    benign_vector, benign_count = sum_group(lambda agent_name: agent_name not in malicious_agents)
    total_vector = malicious_vector + benign_vector
    finite = bool(
        torch.isfinite(malicious_vector).all().item()
        and torch.isfinite(benign_vector).all().item()
        and torch.isfinite(total_vector).all().item()
    )

    return {
        "malicious_client_count": malicious_count,
        "benign_client_count": benign_count,
        "malicious_aggregate_norm": tensor_l2_norm(malicious_vector),
        "benign_aggregate_norm": tensor_l2_norm(benign_vector),
        "total_aggregate_norm": tensor_l2_norm(total_vector),
        "applied_total_update_norm": tensor_l2_norm(total_vector * eta_over_no_models),
        "malicious_to_benign_cosine": tensor_vector_cosine(malicious_vector, benign_vector),
        "malicious_to_total_cosine": tensor_vector_cosine(malicious_vector, total_vector),
        "benign_to_total_cosine": tensor_vector_cosine(benign_vector, total_vector),
        "eta_over_no_models": eta_over_no_models,
        "finite": finite,
    }


def trigger_projection_metrics(
    updates_by_agent,
    malicious_agents,
    trigger_reference_update,
    scope="whole_model",
    projection_reference="source_trigger_target_gradient",
    parameter_names=None,
):
    """Project malicious/benign/total aggregates onto a trigger-relevant reference direction."""
    agent_items = list(updates_by_agent.items())
    malicious_agents = set(malicious_agents)
    if torch.is_tensor(trigger_reference_update):
        if parameter_names is None:
            raise ValueError("parameter_names is required when trigger_reference_update is a vector")
        names = list(parameter_names)
        reference_vector = trigger_reference_update.detach().double().contiguous().view(-1).cpu()
    else:
        names = parameter_names or parameter_update_names(trigger_reference_update)
        reference_vector = flatten_update_for_parameter_names(trigger_reference_update, names)
    template = torch.zeros_like(reference_vector)

    def sum_group(agent_predicate):
        vector = template.clone()
        count = 0
        for agent_name, update_dict in agent_items:
            if agent_predicate(agent_name):
                vector = vector + flatten_update_for_parameter_names(update_dict, names)
                count += 1
        return vector, count

    malicious_vector, _ = sum_group(lambda agent_name: agent_name in malicious_agents)
    benign_vector, _ = sum_group(lambda agent_name: agent_name not in malicious_agents)
    total_vector = malicious_vector + benign_vector
    reference_norm = reference_vector.double().norm()
    reference_norm_value = float(reference_norm.item()) if reference_vector.numel() else 0.0
    if reference_norm_value == 0.0:
        reference_unit = template.clone()
    else:
        reference_unit = reference_vector.double() / reference_norm

    def projection(vector):
        if vector.numel() == 0:
            return 0.0
        value = float(torch.dot(vector.double(), reference_unit).item())
        return value if math.isfinite(value) else float("nan")

    def projection_ratio(value):
        if reference_norm_value == 0.0:
            return 0.0
        ratio = value / reference_norm_value
        return ratio if math.isfinite(ratio) else float("nan")

    malicious_projection = projection(malicious_vector)
    benign_projection = projection(benign_vector)
    total_projection = projection(total_vector)
    finite = bool(
        torch.isfinite(reference_vector).all().item()
        and torch.isfinite(malicious_vector).all().item()
        and torch.isfinite(benign_vector).all().item()
        and torch.isfinite(total_vector).all().item()
    )

    return {
        "scope": scope,
        "projection_reference": projection_reference,
        "malicious_projection": malicious_projection,
        "benign_projection": benign_projection,
        "total_projection": total_projection,
        "malicious_projection_ratio": projection_ratio(malicious_projection),
        "benign_projection_ratio": projection_ratio(benign_projection),
        "total_projection_ratio": projection_ratio(total_projection),
        "malicious_cosine": tensor_vector_cosine(malicious_vector, reference_vector),
        "benign_cosine": tensor_vector_cosine(benign_vector, reference_vector),
        "total_cosine": tensor_vector_cosine(total_vector, reference_vector),
        "finite": finite,
    }


def virtual_actual_gap_metric(
    virtual_backdoor_loss,
    actual_post_aggregation_backdoor_loss,
    virtual_asr_proxy="",
    actual_global_asr="",
):
    """Compare APD's virtual loss prediction with actual post-aggregation backdoor loss."""
    virtual_loss = float(virtual_backdoor_loss)
    actual_loss = float(actual_post_aggregation_backdoor_loss)
    gap = actual_loss - virtual_loss
    values_to_check = [virtual_loss, actual_loss, gap]
    for optional in (virtual_asr_proxy, actual_global_asr):
        if optional != "":
            values_to_check.append(float(optional))
    finite = all(math.isfinite(value) for value in values_to_check)
    return {
        "virtual_backdoor_loss": virtual_loss,
        "actual_post_aggregation_backdoor_loss": actual_loss,
        "virtual_actual_loss_gap": gap if math.isfinite(gap) else float("nan"),
        "virtual_asr_proxy": virtual_asr_proxy,
        "actual_global_asr": actual_global_asr,
        "finite": finite,
    }


def flatten_update_like_parameters(update_dict, model):
    values = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name not in update_dict:
            raise KeyError(f"Missing update tensor for parameter: {name}")
        values.append(update_dict[name].detach().float().contiguous().view(-1).cpu())
    if not values:
        return torch.empty(0)
    return torch.cat(values)
