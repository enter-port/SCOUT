"""State autoencoder E_s / D_s (scout_design.md §2, §3).

Learns a state <-> latent mapping so the VIB dynamics operates on a compact,
reconstructible ``s̄`` instead of raw states. The AE-reconstruction term of the
joint loss (``scout_vib.py``) uses ``D_s(E_s(S))`` as an anti-collapse anchor:
because the next-latent target ``s̄_{t+1}=E_s(S_{t+1})`` is *not* detached, the
dynamics pull would otherwise drift ``E_s`` toward a trivial constant; the
anchor pins "can be decoded back to S".

Stage-1 = low_dim MLP. Stage-2 image path is a drop-in via
``StateAE.from_config('image', ...)`` -- see :class:`StateCnnAE` placeholder
and the **【image 接口点】** note below.
"""

import torch.nn as nn

from scout.model.mlp import EncoderMLP


class StateAE(nn.Module):
    """Abstract ``E_s: state -> s̄`` / ``D_s: s̄ -> Ŝ`` interface.

    Subclasses implement :meth:`encode` and :meth:`decode` with tensor IO of
    shape ``(B, state_dim)`` <-> ``(B, latent_dim)``. The factory
    :meth:`from_config` selects the concrete class by modality.
    """

    def encode(self, s):
        raise NotImplementedError

    def decode(self, s_bar):
        raise NotImplementedError

    def forward(self, s):
        """Default = round-trip (convenience for AE-only sanity checks)."""
        return self.decode(self.encode(s))

    @staticmethod
    def from_config(modality, state_dim, latent_dim=32, hidden_dim=128):
        """Factory by modality string.

        - ``'low_dim'`` -> :class:`StateMLPAE` (stage-1).
        - ``'image'``   -> :class:`StateCnnAE` (stage-2, not yet implemented).

        The VIB encoder/decoder, guidance, and cost modules only consume the
        ``s̄`` vector, so they are modality-agnostic -- swapping modality here
        is the only change needed for the image path (scout_design.md §6).
        """
        if modality == "low_dim":
            return StateMLPAE(state_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        if modality == "image":
            # **【image 接口点】** stage-2: implement StateCnnAE here (per-view
            # ResNet encoder à la LPB ResNetEncoder + a decode strategy TBD per
            # scout_design.md §6 option (i)/(ii)). Keep encode/decode signatures
            # identical so the rest of the pipeline is untouched.
            return StateCnnAE(state_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        raise ValueError(f"unknown StateAE modality: {modality!r}")


class StateMLPAE(StateAE):
    """Low-dim state AE: both encoder and decoder are :class:`EncoderMLP`.

    ``EncoderMLP`` is the SOE block (Linear->ReLU->[Linear->ReLU]*L->Linear);
    we instantiate two independent ones (no weight sharing).
    """

    def __init__(self, state_dim, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = EncoderMLP(state_dim, latent_dim, hidden_dim=hidden_dim)
        self.decoder = EncoderMLP(latent_dim, state_dim, hidden_dim=hidden_dim)

    def encode(self, s):
        return self.encoder(s)

    def decode(self, s_bar):
        return self.decoder(s_bar)


class StateCnnAE(StateAE):
    """**【image 接口点】** stage-2 placeholder -- not implemented yet.

    Reserved so :meth:`StateAE.from_config` can dispatch on ``'image'`` without
    touching the rest of the pipeline. The stage-2 build will replace ``raise``
    with a CNN encoder (+ a decode policy per scout_design.md §6) while keeping
    :meth:`encode`/:meth:`decode` signatures.
    """

    def __init__(self, state_dim, latent_dim=32, hidden_dim=128):
        super().__init__()
        raise NotImplementedError(
            "StateCnnAE (image E_s/D_s) is stage-2 work -- see scout_design.md §6. "
            "Stage-1 is low_dim only; use StateMLPAE via from_config('low_dim', ...)."
        )
