#!/usr/bin/env python3

import hashlib
import random
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import numpy as np
import torch


_STREAM_NAMES = ("classifier_init", "prompt_init", "data_order")


def derive_seed(master_seed: Optional[int], stream_name: str) -> Optional[int]:
    if master_seed is None:
        return None
    payload = "vpt-repro-v1|{}|{}".format(int(master_seed), str(stream_name)).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)
    return value % ((1 << 63) - 1)


def seed_streams(master_seed: Optional[int]) -> Dict[str, Optional[int]]:
    return {
        name: derive_seed(master_seed, name)
        for name in _STREAM_NAMES
    }


def rank_runtime_seed(master_seed: Optional[int], rank: int) -> Optional[int]:
    """Return the deterministic training-time seed for one distributed rank."""
    return derive_seed(master_seed, "rank_runtime:{}".format(int(rank)))


def apply_rank_runtime_seed(master_seed: Optional[int], rank: int) -> Optional[int]:
    """Seed runtime stochasticity after shared model initialization is complete."""
    seed = rank_runtime_seed(master_seed, rank)
    if seed is None:
        return None
    random.seed(int(seed))
    np.random.seed(int(seed) % (1 << 32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return seed


def make_torch_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


@contextmanager
def isolated_torch_cpu_seed(seed: Optional[int]) -> Iterator[None]:
    if seed is None:
        yield
        return
    rng_state = torch.get_rng_state()
    torch.default_generator.manual_seed(int(seed))
    try:
        yield
    finally:
        torch.set_rng_state(rng_state)
