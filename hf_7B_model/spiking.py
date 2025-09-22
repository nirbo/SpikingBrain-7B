# -*- coding: utf-8 -*-

"""Utility helpers for optional SpikingBrain activation modeling.

These routines mirror the dynamic spike encoding and symmetric quantization
strategy used in the W8ASpike reference so that the HuggingFace variant can
simulate spike-based computation without depending on the quantized model.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - optional dependency
    from W8ASpike.Int2Spike.neuron import SpikeCountBitwiseNode, spike_fake_quant

    _SPIKE_BACKEND_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback when Int2Spike is absent
    SpikeCountBitwiseNode = None  # type: ignore
    spike_fake_quant = None  # type: ignore
    _SPIKE_BACKEND_AVAILABLE = False


def dynamic_spikes(x: torch.Tensor, scale_factor: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert activations into integer spike counts with adaptive thresholds.

    Args:
        x: Activation tensor shaped [..., head_dim].
        scale_factor: Divisor used to derive the adaptive threshold from the mean
            absolute activation magnitude (matches the technical report guidance).

    Returns:
        spikes_int: Integer spike counts after rounding and optional fake spike encoding.
        vth: The adaptive threshold used to normalize the activations.
    """

    vth = x.abs().mean(dim=-1, keepdim=True).float() / scale_factor
    vth = vth.clamp(min=1e-5, max=1e4)
    spikes_int = (x / vth).round()

    if _SPIKE_BACKEND_AVAILABLE:
        lif = SpikeCountBitwiseNode(is_bidirectional=True)
        spikes_int = spike_fake_quant(spikes_int, lif_quantizer=lif)

    return spikes_int, vth


def quantize_sym(x: torch.Tensor, bitwidth: int = 8) -> torch.Tensor:
    """Apply symmetric per-vector quantization to mimic spiking sparsity.

    Args:
        x: Tensor to quantize.
        bitwidth: Number of bits for the symmetric integer range.

    Returns:
        The quantized tensor cast back to the original dtype.
    """

    q_max = (1 << (bitwidth - 1)) - 1
    q_min = -(1 << (bitwidth - 1))
    orig_dtype = x.dtype
    scale = x.abs().amax(dim=-1, keepdim=True).float() / max(q_max, 1)
    scale = scale.clamp(min=1e-5, max=1e4)

    x_int = (x / scale).round().clamp(q_min, q_max)
    return (x_int * scale).to(orig_dtype)
