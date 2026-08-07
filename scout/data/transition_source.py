"""Transition data source abstraction + in-memory ``ReplayBuffer``.

The core training unit is a transition ``(S_t, A_t, S_{t+1})`` where ``S_t`` is
the already-concatenated state vector (see scout_design.md §3: 数据 loader
解耦). Backends only need to produce these triples; training code talks to
them solely through :class:`TransitionSource`.

Dict keys are ``"S_t"``, ``"A_t"``, ``"S_tp1"`` (tp1 == t-plus-1 == S_{t+1}).
"""

from abc import ABC, abstractmethod
import torch

from scout.data.running_stats import RunningStats


class TransitionSource(ABC):
    """Abstract transition supplier.

    Backends: :class:`ReplayBuffer` (in-memory, online) and
    :class:`scout.data.robomimic_lowdim.RobomimicLowdimSource` (offline file).
    """

    @abstractmethod
    def sample(self, batch_size: int) -> dict:
        """Random batch ``{"S_t", "A_t", "S_tp1"}``, each ``(batch_size, dim)``."""

    @abstractmethod
    def add(self, transitions: dict) -> None:
        """Online write-back of new transitions (self-improvement loop entry)."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def stats(self) -> dict:
        """``{"S_t", "A_t", "S_tp1"} -> RunningStats`` over the current contents."""


class ReplayBuffer(TransitionSource):
    """Fixed-capacity circular buffer of (S_t, A_t, S_{t+1}) triples on CPU.

    States and actions are stored as independent rows -- the buffer is agnostic
    to trajectory structure, which keeps the add/sample path simple and matches
    the online self-improvement write-back (scout_design.md §3, §5).
    """

    def __init__(self, state_dim, action_dim, capacity=int(1e6), dtype=torch.float32):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.dtype = dtype

        self.S_t = torch.zeros(self.capacity, self.state_dim, dtype=dtype)
        self.A_t = torch.zeros(self.capacity, self.action_dim, dtype=dtype)
        self.S_tp1 = torch.zeros(self.capacity, self.state_dim, dtype=dtype)

        self.ptr = 0      # next write index
        self.size = 0     # filled count (<= capacity)

    def add(self, transitions: dict) -> None:
        for key in ("S_t", "A_t", "S_tp1"):
            assert key in transitions, f"missing key {key}"
        s_t = torch.as_tensor(transitions["S_t"], dtype=self.dtype)
        a_t = torch.as_tensor(transitions["A_t"], dtype=self.dtype)
        s_tp1 = torch.as_tensor(transitions["S_tp1"], dtype=self.dtype)
        assert s_t.shape[1:] == (self.state_dim,), f"S_t dim mismatch: {s_t.shape}"
        assert a_t.shape[1:] == (self.action_dim,), f"A_t dim mismatch: {a_t.shape}"
        assert s_tp1.shape[1:] == (self.state_dim,), f"S_tp1 dim mismatch: {s_tp1.shape}"
        n = s_t.shape[0]
        assert n == a_t.shape[0] == s_tp1.shape[0], "mismatched batch dims"

        # circular write with single wrap (add batches << capacity in practice)
        idx = (torch.arange(n) + self.ptr) % self.capacity
        self.S_t[idx] = s_t
        self.A_t[idx] = a_t
        self.S_tp1[idx] = s_tp1
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> dict:
        assert self.size > 0, "sample from empty buffer"
        idx = torch.randint(0, self.size, (batch_size,))
        return {
            "S_t": self.S_t[idx],
            "A_t": self.A_t[idx],
            "S_tp1": self.S_tp1[idx],
        }

    def stats(self) -> dict:
        if self.size == 0:
            raise RuntimeError("stats() on empty buffer")
        s_stats = RunningStats(self.state_dim); s_stats.update(self.S_t[: self.size])
        a_stats = RunningStats(self.action_dim); a_stats.update(self.A_t[: self.size])
        sp_stats = RunningStats(self.state_dim); sp_stats.update(self.S_tp1[: self.size])
        return {"S_t": s_stats, "A_t": a_stats, "S_tp1": sp_stats}

    def __len__(self) -> int:
        return self.size
