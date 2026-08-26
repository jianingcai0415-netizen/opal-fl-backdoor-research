import math

import torch

from .signal_trace import (
    flatten_update_for_parameter_names,
    tensor_l2_norm,
    tensor_vector_cosine,
)


def _copy_update_dict(update_dict):
    return {
        name: tensor.detach().clone() if torch.is_tensor(tensor) else tensor
        for name, tensor in update_dict.items()
    }


def _projection_ratio(vector, descent_vector):
    reference_norm = tensor_l2_norm(descent_vector)
    if reference_norm == 0.0:
        return 0.0
    descent_unit = descent_vector.double() / reference_norm
    projection = float(torch.dot(vector.double(), descent_unit).item())
    if not math.isfinite(projection):
        return float("nan")
    ratio = projection / reference_norm
    return ratio if math.isfinite(ratio) else float("nan")


def select_writein_parameter_names(parameter_names, scope):
    scope = str(scope or "full")
    names = list(parameter_names or [])
    if scope == "full":
        return names
    if scope == "layer4_linear":
        return [
            name for name in names
            if name.startswith("layer4.") or name.startswith("linear.")
        ]
    if scope == "linear":
        return [name for name in names if name.startswith("linear.")]
    raise ValueError(f"Unsupported stage5r_writein_scope: {scope}")


def slice_flat_vector_for_parameter_names(
    vector,
    all_parameter_names,
    selected_parameter_names,
    update_dict,
):
    source = vector.detach().double().contiguous().view(-1).cpu()
    selected = set(selected_parameter_names or [])
    chunks = []
    offset = 0
    for name in list(all_parameter_names or []):
        if name not in update_dict:
            raise KeyError(f"Missing update tensor for parameter: {name}")
        count = update_dict[name].numel()
        next_offset = offset + count
        if next_offset > source.numel():
            raise ValueError("Source vector is shorter than parameter shapes.")
        if name in selected:
            chunks.append(source[offset:next_offset])
        offset = next_offset
    if offset != source.numel():
        raise ValueError("Source vector length does not match parameter shapes.")
    if not chunks:
        return torch.empty(0, dtype=torch.double)
    return torch.cat(chunks)


def _build_stats(
    alpha,
    pre_vector,
    post_vector,
    source_loss_grad_vector,
    applied,
    target_vector=None,
    retention_vector=None,
    retention_gamma=0.0,
    retention_compensation_applied=False,
    retention_memory_vector=None,
    retention_memory_weight=0.0,
    retention_memory_applied=False,
):
    source_vector = source_loss_grad_vector.detach().double().contiguous().view(-1).cpu()
    descent_vector = -source_vector
    target_vector = (
        target_vector.detach().double().contiguous().view(-1).cpu()
        if target_vector is not None
        else descent_vector
    )
    retention_vector = (
        retention_vector.detach().double().contiguous().view(-1).cpu()
        if retention_vector is not None
        else torch.empty(0, dtype=torch.double)
    )
    retention_memory_vector = (
        retention_memory_vector.detach().double().contiguous().view(-1).cpu()
        if retention_memory_vector is not None
        else torch.empty(0, dtype=torch.double)
    )
    pre_target_cosine = tensor_vector_cosine(pre_vector, target_vector)
    post_target_cosine = tensor_vector_cosine(post_vector, target_vector)
    values = {
        "alpha": float(alpha),
        "applied": bool(applied),
        "pre_update_norm": tensor_l2_norm(pre_vector),
        "post_update_norm": tensor_l2_norm(post_vector),
        "source_grad_norm": tensor_l2_norm(source_vector),
        "pre_descent_cosine": tensor_vector_cosine(pre_vector, descent_vector),
        "post_descent_cosine": tensor_vector_cosine(post_vector, descent_vector),
        "pre_descent_projection_ratio": _projection_ratio(pre_vector, descent_vector),
        "post_descent_projection_ratio": _projection_ratio(post_vector, descent_vector),
        "retention_gamma": float(retention_gamma),
        "retention_vector_norm": tensor_l2_norm(retention_vector),
        "retention_compensation_applied": bool(retention_compensation_applied),
        "target_vector_norm": tensor_l2_norm(target_vector),
        "pre_target_cosine": pre_target_cosine,
        "post_target_cosine": post_target_cosine,
        "post_minus_pre_target_cosine": (
            post_target_cosine - pre_target_cosine
            if math.isfinite(post_target_cosine) and math.isfinite(pre_target_cosine)
            else float("nan")
        ),
        "retention_memory_weight": float(retention_memory_weight),
        "retention_memory_norm": tensor_l2_norm(retention_memory_vector),
        "retention_memory_applied": bool(retention_memory_applied),
    }
    finite_scalars = all(
        math.isfinite(float(value))
        for value in values.values()
        if isinstance(value, (int, float))
    )
    values["finite"] = bool(
        finite_scalars
        and torch.isfinite(pre_vector).all().item()
        and torch.isfinite(post_vector).all().item()
        and torch.isfinite(source_vector).all().item()
        and torch.isfinite(target_vector).all().item()
        and torch.isfinite(retention_vector).all().item()
        and torch.isfinite(retention_memory_vector).all().item()
    )
    return values


def apply_submission_trigger_writein(
    update_dict,
    parameter_names,
    source_loss_grad_vector,
    alpha,
    retention_vector=None,
    retention_gamma=0.0,
    retention_memory_vector=None,
    retention_memory_weight=0.0,
    eps=1e-12,
):
    """Rotate actual submitted parameter updates toward source-trigger loss descent.

    Buffers and tensors outside ``parameter_names`` are copied unchanged. ``alpha=0``
    is an exact no-op in value space.
    """
    alpha = float(alpha)
    retention_gamma = float(retention_gamma)
    retention_memory_weight = float(retention_memory_weight)
    names = list(parameter_names or [])
    transformed = _copy_update_dict(update_dict)
    if not names:
        empty = torch.empty(0, dtype=torch.double)
        return transformed, _build_stats(
            alpha,
            empty,
            empty,
            empty,
            applied=False,
            retention_gamma=retention_gamma,
            retention_memory_weight=retention_memory_weight,
        )

    pre_vector = flatten_update_for_parameter_names(update_dict, names)
    source_vector = source_loss_grad_vector.detach().double().contiguous().view(-1).cpu()
    if pre_vector.numel() != source_vector.numel():
        raise ValueError(
            f"Submission write-in vector sizes must match: {pre_vector.numel()} != {source_vector.numel()}"
        )

    if alpha <= 0.0:
        return transformed, _build_stats(
            alpha,
            pre_vector,
            pre_vector,
            source_vector,
            applied=False,
            retention_gamma=retention_gamma,
            retention_memory_weight=retention_memory_weight,
        )

    pre_norm = pre_vector.double().norm()
    source_norm = source_vector.double().norm()
    pre_norm_value = float(pre_norm.item()) if pre_vector.numel() else 0.0
    source_norm_value = float(source_norm.item()) if source_vector.numel() else 0.0
    if (
        pre_norm_value <= eps
        or source_norm_value <= eps
        or not math.isfinite(pre_norm_value)
        or not math.isfinite(source_norm_value)
        or not bool(torch.isfinite(pre_vector).all().item())
        or not bool(torch.isfinite(source_vector).all().item())
    ):
        return transformed, _build_stats(
            alpha,
            pre_vector,
            pre_vector,
            source_vector,
            applied=False,
            retention_gamma=retention_gamma,
            retention_memory_weight=retention_memory_weight,
        )

    pre_unit = pre_vector.double() / pre_norm
    descent_vector = -source_vector.double()
    target_vector = descent_vector
    retention_vector_for_stats = None
    retention_compensation_applied = False
    retention_memory_vector_for_stats = None
    retention_memory_applied = False
    if retention_gamma > 0.0 and retention_vector is not None:
        candidate_retention = retention_vector.detach().double().contiguous().view(-1).cpu()
        retention_vector_for_stats = candidate_retention
        candidate_norm_value = float(candidate_retention.norm().item()) if candidate_retention.numel() else 0.0
        if (
            candidate_retention.numel() == source_vector.numel()
            and candidate_norm_value > eps
            and math.isfinite(candidate_norm_value)
            and bool(torch.isfinite(candidate_retention).all().item())
        ):
            candidate_target = descent_vector - retention_gamma * candidate_retention
            candidate_target_norm_value = (
                float(candidate_target.norm().item()) if candidate_target.numel() else 0.0
            )
            if (
                candidate_target_norm_value > eps
                and math.isfinite(candidate_target_norm_value)
                and bool(torch.isfinite(candidate_target).all().item())
            ):
                target_vector = candidate_target
                retention_compensation_applied = True

    if retention_memory_weight > 0.0 and retention_memory_vector is not None:
        candidate_memory = retention_memory_vector.detach().double().contiguous().view(-1).cpu()
        retention_memory_vector_for_stats = candidate_memory
        candidate_memory_norm_value = float(candidate_memory.norm().item()) if candidate_memory.numel() else 0.0
        if (
            candidate_memory.numel() == source_vector.numel()
            and candidate_memory_norm_value > eps
            and math.isfinite(candidate_memory_norm_value)
            and bool(torch.isfinite(candidate_memory).all().item())
        ):
            candidate_target = target_vector + retention_memory_weight * candidate_memory
            candidate_target_norm_value = (
                float(candidate_target.norm().item()) if candidate_target.numel() else 0.0
            )
            if (
                candidate_target_norm_value > eps
                and math.isfinite(candidate_target_norm_value)
                and bool(torch.isfinite(candidate_target).all().item())
            ):
                target_vector = candidate_target
                retention_memory_applied = True

    target_norm = target_vector.double().norm()
    target_norm_value = float(target_norm.item()) if target_vector.numel() else 0.0
    if target_norm_value <= eps or not math.isfinite(target_norm_value):
        return transformed, _build_stats(
            alpha,
            pre_vector,
            pre_vector,
            source_vector,
            applied=False,
            retention_vector=retention_vector_for_stats,
            retention_gamma=retention_gamma,
            retention_compensation_applied=retention_compensation_applied,
            retention_memory_vector=retention_memory_vector_for_stats,
            retention_memory_weight=retention_memory_weight,
            retention_memory_applied=retention_memory_applied,
        )

    target_unit = target_vector.double() / target_norm
    mixed = (1.0 - alpha) * pre_unit + alpha * target_unit
    mixed_norm = mixed.norm()
    mixed_norm_value = float(mixed_norm.item())
    if mixed_norm_value <= eps or not math.isfinite(mixed_norm_value):
        return transformed, _build_stats(
            alpha,
            pre_vector,
            pre_vector,
            source_vector,
            applied=False,
            target_vector=target_vector,
            retention_vector=retention_vector_for_stats,
            retention_gamma=retention_gamma,
            retention_compensation_applied=retention_compensation_applied,
            retention_memory_vector=retention_memory_vector_for_stats,
            retention_memory_weight=retention_memory_weight,
            retention_memory_applied=retention_memory_applied,
        )

    post_vector = (mixed / mixed_norm) * pre_norm
    offset = 0
    for name in names:
        original = update_dict[name]
        count = original.numel()
        transformed[name] = post_vector[offset:offset + count].view_as(original).to(
            device=original.device,
            dtype=original.dtype,
        )
        offset += count

    return transformed, _build_stats(
        alpha,
        pre_vector,
        post_vector,
        source_vector,
        applied=True,
        target_vector=target_vector,
        retention_vector=retention_vector_for_stats,
        retention_gamma=retention_gamma,
        retention_compensation_applied=retention_compensation_applied,
        retention_memory_vector=retention_memory_vector_for_stats,
        retention_memory_weight=retention_memory_weight,
        retention_memory_applied=retention_memory_applied,
    )
