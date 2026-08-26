import hashlib
from collections import OrderedDict

import torch


class PoisonMemory:
    """Persistent per-agent clean-label perturbation memory."""

    def __init__(self, epsilon, update_rate=1.0, max_size_per_client=None):
        epsilon = float(epsilon)
        update_rate = float(update_rate)
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if not 0.0 <= update_rate <= 1.0:
            raise ValueError("update_rate must be in [0, 1]")
        if max_size_per_client is not None and int(max_size_per_client) <= 0:
            raise ValueError("max_size_per_client must be positive when set")

        self.epsilon = epsilon
        self.update_rate = update_rate
        self.max_size_per_client = (
            None if max_size_per_client is None else int(max_size_per_client)
        )
        self._store = {}
        self._last_keys = {}

    def get(self, agent, indices, images, targets=None):
        keys = self._make_keys(agent, indices, images, targets)
        self._last_keys[agent] = keys
        result = torch.zeros_like(images)
        store = self._agent_store(agent)

        for row_idx, key in enumerate(keys):
            if key not in store:
                continue
            stored_delta = store[key]
            result[row_idx].copy_(stored_delta.to(device=images.device, dtype=images.dtype))
            store.move_to_end(key)

        return result

    def update(self, agent, indices, new_delta):
        if indices is None:
            if agent not in self._last_keys:
                raise ValueError("indices are required unless get() was called first for this agent")
            keys = self._last_keys[agent]
        else:
            keys = self._index_keys(indices)
        if len(keys) != len(new_delta):
            raise ValueError("number of memory keys must match new_delta batch size")

        store = self._agent_store(agent)
        clipped_delta = new_delta.detach().cpu().clamp(min=-self.epsilon, max=self.epsilon)
        for row_idx, key in enumerate(keys):
            candidate = clipped_delta[row_idx].clone()
            if key in store:
                previous = store[key].to(dtype=candidate.dtype)
                candidate = (1.0 - self.update_rate) * previous + self.update_rate * candidate
                candidate = candidate.clamp(min=-self.epsilon, max=self.epsilon)
            store[key] = candidate
            store.move_to_end(key)
            self._evict_if_needed(store)

    def _agent_store(self, agent):
        if agent not in self._store:
            self._store[agent] = OrderedDict()
        return self._store[agent]

    def _make_keys(self, agent, indices, images, targets):
        if indices is not None:
            return self._index_keys(indices)
        return self._hash_keys(images, targets)

    @staticmethod
    def _index_keys(indices):
        if torch.is_tensor(indices):
            values = indices.detach().cpu().view(-1).tolist()
        else:
            values = list(indices)
        return [("index", int(value)) for value in values]

    @staticmethod
    def _hash_keys(images, targets):
        image_cpu = images.detach().cpu().contiguous()
        target_cpu = None
        if targets is not None:
            target_cpu = targets.detach().cpu().view(-1)

        keys = []
        for row_idx in range(len(image_cpu)):
            digest = hashlib.sha1()
            sample = image_cpu[row_idx].contiguous()
            digest.update(str(tuple(sample.shape)).encode("utf-8"))
            digest.update(str(sample.dtype).encode("utf-8"))
            digest.update(sample.numpy().tobytes())
            if target_cpu is not None:
                digest.update(str(int(target_cpu[row_idx].item())).encode("utf-8"))
            keys.append(("hash", digest.hexdigest()))
        return keys

    def _evict_if_needed(self, store):
        if self.max_size_per_client is None:
            return
        while len(store) > self.max_size_per_client:
            store.popitem(last=False)
