"""Losses for 2-stage controllable restoration training."""

from .controllable_losses import ControllableRestorationLoss, SpectralPatchDiscriminator

__all__ = ["ControllableRestorationLoss", "SpectralPatchDiscriminator"]
