import math

import torch
import torch.nn.functional as F
from torch.func import functional_call


def build_virtual_fedavg_state_dict(model, malicious_update, benign_drift=None, eta_over_no_models=1.0):
    """Return a state dict for theta + eta/K * (u_malicious + u_benign)."""
    eta_over_no_models = float(eta_over_no_models)
    benign_drift = benign_drift or {}
    virtual_state = {}

    for name, parameter in model.named_parameters():
        malicious = malicious_update.get(name)
        benign = benign_drift.get(name)
        if malicious is None:
            malicious = torch.zeros_like(parameter)
        if benign is None:
            benign = torch.zeros_like(parameter)
        virtual_state[name] = parameter + eta_over_no_models * (
            malicious.to(device=parameter.device, dtype=parameter.dtype)
            + benign.to(device=parameter.device, dtype=parameter.dtype)
        )

    for name, buffer in model.named_buffers():
        virtual_state[name] = buffer

    return virtual_state


def virtual_backdoor_loss(
    model,
    source_trigger_images,
    target_labels,
    malicious_update,
    benign_drift=None,
    eta_over_no_models=1.0,
):
    virtual_state = build_virtual_fedavg_state_dict(
        model,
        malicious_update=malicious_update,
        benign_drift=benign_drift,
        eta_over_no_models=eta_over_no_models,
    )
    logits = functional_call(model, virtual_state, (source_trigger_images,))
    loss = F.cross_entropy(logits, target_labels)
    stats = {
        "virtual_backdoor_loss": float(loss.detach().item()),
        "virtual_batch_size": int(len(source_trigger_images)),
        "malicious_update_norm": _update_norm(malicious_update),
        "benign_drift_norm": _update_norm(benign_drift or {}),
        "eta_over_no_models": eta_over_no_models,
        "finite": bool(math.isfinite(float(loss.detach().item()))),
    }
    return loss, stats


def build_poison_gradient_update(model, poison_images, poison_labels, local_lr_proxy=1.0):
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    logits = model(poison_images)
    poison_loss = F.cross_entropy(logits, poison_labels)
    gradients = torch.autograd.grad(
        poison_loss,
        [parameter for _, parameter in named_parameters],
        retain_graph=True,
        create_graph=True,
        allow_unused=True,
    )

    update = {}
    for (name, parameter), gradient in zip(named_parameters, gradients):
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        update[name] = -float(local_lr_proxy) * gradient
    return update, poison_loss


def apd_poison_objective(
    model,
    poison_images,
    poison_labels,
    source_trigger_images,
    target_labels,
    local_lr_proxy=1.0,
    eta_over_no_models=1.0,
    benign_drift=None,
    update_norm_target=None,
    trigger_projection_lambda=0.0,
):
    malicious_update, poison_loss = build_poison_gradient_update(
        model,
        poison_images=poison_images,
        poison_labels=poison_labels,
        local_lr_proxy=local_lr_proxy,
    )
    raw_malicious_update_norm = _update_norm(malicious_update)
    if update_norm_target is not None:
        malicious_update = scale_update_dict_to_norm(
            malicious_update,
            target_norm=float(update_norm_target),
        )
    loss, stats = virtual_backdoor_loss(
        model,
        source_trigger_images=source_trigger_images,
        target_labels=target_labels,
        malicious_update=malicious_update,
        benign_drift=benign_drift,
        eta_over_no_models=eta_over_no_models,
    )
    trigger_projection_lambda = float(trigger_projection_lambda)
    if trigger_projection_lambda > 0.0:
        source_loss_gradient = source_trigger_loss_gradient(
            model,
            source_trigger_images=source_trigger_images,
            target_labels=target_labels,
        )
        descent_reward = trigger_descent_projection_reward(
            malicious_update,
            source_loss_gradient,
        )
        projection_loss_component = -trigger_projection_lambda * descent_reward
        loss = loss + projection_loss_component
        stats["trigger_descent_projection_reward"] = float(descent_reward.detach().item())
        stats["trigger_projection_lambda"] = trigger_projection_lambda
        stats["trigger_descent_projection_loss_component"] = float(
            projection_loss_component.detach().item()
        )
    stats["poison_surrogate_loss"] = float(poison_loss.detach().item())
    stats["local_lr_proxy"] = float(local_lr_proxy)
    stats["raw_malicious_update_norm"] = raw_malicious_update_norm
    stats["update_norm_target"] = update_norm_target
    stats["finite"] = bool(
        stats["finite"]
        and math.isfinite(stats["poison_surrogate_loss"])
        and math.isfinite(stats["raw_malicious_update_norm"])
        and math.isfinite(stats["malicious_update_norm"])
        and math.isfinite(stats.get("trigger_descent_projection_reward", 0.0))
        and math.isfinite(stats.get("trigger_descent_projection_loss_component", 0.0))
    )
    return loss, stats


def source_trigger_loss_gradient(model, source_trigger_images, target_labels):
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    logits = model(source_trigger_images)
    source_loss = F.cross_entropy(logits, target_labels)
    gradients = torch.autograd.grad(
        source_loss,
        [parameter for _, parameter in named_parameters],
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    loss_gradient = {}
    for (name, parameter), gradient in zip(named_parameters, gradients):
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        loss_gradient[name] = gradient.detach()
    return loss_gradient


def trigger_descent_projection_reward(update_tensors, loss_gradient_tensors, eps=1e-12):
    update_flat, loss_gradient_flat = _aligned_flatten_tensors(
        update_tensors,
        loss_gradient_tensors,
    )
    if update_flat is None or loss_gradient_flat is None:
        return torch.tensor(0.0)

    update_flat = update_flat.float()
    loss_gradient_flat = loss_gradient_flat.to(device=update_flat.device).float()
    loss_gradient_norm = loss_gradient_flat.norm()
    if not bool(torch.isfinite(loss_gradient_norm).detach().cpu().item()):
        return update_flat.sum() * 0.0
    if float(loss_gradient_norm.detach().cpu().item()) <= eps:
        return update_flat.sum() * 0.0

    descent_direction = -loss_gradient_flat / loss_gradient_norm
    projection = torch.dot(update_flat, descent_direction)
    return torch.clamp(projection, min=0.0)


def scale_update_dict_to_norm(update_dict, target_norm, eps=1e-12):
    current_norm = _update_norm_tensor(update_dict)
    if not bool(torch.isfinite(current_norm).detach().cpu().item()):
        return update_dict
    if float(current_norm.detach().cpu().item()) <= eps:
        return update_dict

    scale = float(target_norm) / current_norm
    return {
        name: tensor * scale.to(device=tensor.device, dtype=tensor.dtype)
        for name, tensor in update_dict.items()
    }


def _aligned_flatten_tensors(update_tensors, reference_tensors):
    update_parts = []
    reference_parts = []

    if isinstance(update_tensors, dict) and isinstance(reference_tensors, dict):
        names = [name for name in update_tensors.keys() if name in reference_tensors]
        for name in names:
            update_tensor = update_tensors[name]
            reference_tensor = reference_tensors[name]
            if reference_tensor is None:
                reference_tensor = torch.zeros_like(update_tensor)
            reference_tensor = reference_tensor.to(
                device=update_tensor.device,
                dtype=update_tensor.dtype,
            )
            update_parts.append(update_tensor.reshape(-1))
            reference_parts.append(reference_tensor.reshape(-1))
    else:
        for update_tensor, reference_tensor in zip(update_tensors, reference_tensors):
            if reference_tensor is None:
                reference_tensor = torch.zeros_like(update_tensor)
            reference_tensor = reference_tensor.to(
                device=update_tensor.device,
                dtype=update_tensor.dtype,
            )
            update_parts.append(update_tensor.reshape(-1))
            reference_parts.append(reference_tensor.reshape(-1))

    if not update_parts:
        return None, None
    return torch.cat(update_parts), torch.cat(reference_parts)


def _update_norm_tensor(update_dict):
    total = None
    for tensor in update_dict.values():
        value = tensor.float().pow(2).sum()
        if total is None:
            total = value
        else:
            total = total + value.to(device=total.device)
    if total is None:
        return torch.tensor(0.0)
    return torch.sqrt(total)


def _update_norm(update_dict):
    squared = 0.0
    for tensor in update_dict.values():
        value = tensor.detach().float().norm().item()
        squared += value * value
    return math.sqrt(squared)
