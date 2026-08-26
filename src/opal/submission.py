import math

import torch


BUFFER_NAME_MARKERS = (
    "running_mean",
    "running_var",
    "num_batches_tracked",
)


def _is_parameter_name(name, parameter_names):
    if parameter_names is None:
        return not any(marker in name for marker in BUFFER_NAME_MARKERS)
    return name in set(parameter_names)


def _validate_keys(local_state, reference_state):
    if set(local_state) != set(reference_state):
        missing_local = sorted(set(reference_state) - set(local_state))
        missing_reference = sorted(set(local_state) - set(reference_state))
        raise ValueError(
            f"State dictionaries must have identical keys. "
            f"missing_local={missing_local}, missing_reference={missing_reference}"
        )


def _validate_finite_state(state):
    for name, tensor in state.items():
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite submitted state: {name}")


def _state_l2_norm(state):
    squared_norm = 0.0
    for tensor in state.values():
        if tensor.is_floating_point() or tensor.is_complex():
            squared_norm += tensor.detach().double().abs().pow(2).sum().item()
    return math.sqrt(squared_norm)


def _delta_l2_norm(delta_by_name, selected_names):
    squared_norm = 0.0
    selected_names = set(selected_names)
    for name, tensor in delta_by_name.items():
        if name not in selected_names:
            continue
        if tensor.is_floating_point() or tensor.is_complex():
            squared_norm += tensor.detach().double().abs().pow(2).sum().item()
    return math.sqrt(squared_norm)


def _scaled_tensor(reference, local, scale):
    if local.is_floating_point() or local.is_complex():
        return reference.detach().clone() + (local.detach() - reference.detach()) * float(scale)
    if float(scale).is_integer():
        return reference.detach().clone() + (local.detach() - reference.detach()) * int(scale)
    return local.detach().clone()


def build_submitted_state_dict(
    local_state,
    reference_state,
    mode,
    scale,
    scale_buffers,
    l2_cap,
    parameter_names=None,
):
    _validate_keys(local_state, reference_state)
    _validate_finite_state(local_state)
    _validate_finite_state(reference_state)

    supported_modes = {"full_state_scaled", "parameter_delta_scaled", "bounded_delta"}
    if mode not in supported_modes:
        raise ValueError(f"Unsupported submitted_update_mode: {mode}")

    parameter_names = set(parameter_names) if parameter_names is not None else None
    selected_parameter_names = [
        name for name in local_state
        if _is_parameter_name(name, parameter_names)
    ]
    buffer_names = [name for name in local_state if name not in set(selected_parameter_names)]

    raw_scaled_delta = {}
    for name in selected_parameter_names:
        local = local_state[name]
        reference = reference_state[name]
        if local.is_floating_point() or local.is_complex():
            raw_scaled_delta[name] = (local.detach() - reference.detach()) * float(scale)
        elif float(scale).is_integer():
            raw_scaled_delta[name] = (local.detach() - reference.detach()) * int(scale)
        else:
            raw_scaled_delta[name] = local.detach() - reference.detach()

    clip_ratio = 1.0
    if mode == "bounded_delta" and l2_cap is not None:
        raw_parameter_norm = _delta_l2_norm(raw_scaled_delta, selected_parameter_names)
        if raw_parameter_norm > 0.0:
            clip_ratio = min(1.0, float(l2_cap) / raw_parameter_norm)
        for name in raw_scaled_delta:
            if raw_scaled_delta[name].is_floating_point() or raw_scaled_delta[name].is_complex():
                raw_scaled_delta[name] = raw_scaled_delta[name] * clip_ratio

    submitted = {}
    for name in local_state:
        local = local_state[name]
        reference = reference_state[name]
        is_parameter = name in set(selected_parameter_names)

        if mode == "full_state_scaled":
            if is_parameter or scale_buffers:
                submitted[name] = _scaled_tensor(reference, local, scale)
            else:
                submitted[name] = local.detach().clone()
        elif is_parameter:
            submitted[name] = reference.detach().clone() + raw_scaled_delta[name]
        else:
            submitted[name] = local.detach().clone()

    _validate_finite_state(submitted)
    parameter_update_norm = _delta_l2_norm(
        {name: submitted[name] - reference_state[name] for name in selected_parameter_names},
        selected_parameter_names,
    )
    buffer_update_norm = _delta_l2_norm(
        {name: submitted[name] - reference_state[name] for name in buffer_names},
        buffer_names,
    )
    stats = {
        "mode": mode,
        "scale": float(scale),
        "scale_model_buffers": bool(scale_buffers),
        "submitted_update_l2_cap": "" if l2_cap is None else float(l2_cap),
        "pre_scale_model_norm": _state_l2_norm(local_state),
        "post_scale_model_norm": _state_l2_norm(submitted),
        "parameter_update_norm": parameter_update_norm,
        "buffer_update_norm": buffer_update_norm,
        "clip_ratio": clip_ratio,
        "finite_state": True,
    }
    return submitted, stats
