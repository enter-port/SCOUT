"""Compact snapshots for trajectory writers; policy observations stay untouched."""
import numpy as np


class StorageObservation(dict):
    """One current frame per key, already in the HDF5 writer's storage layout."""


def storage_snapshot(obs):
    # Use exactly the writer's conversion, at collection time instead of finalization.
    from scout.eval.hdf5_writer import _last_frame, _to_storage
    return StorageObservation({key: np.array(_to_storage(key, _last_frame(value)), copy=True)
                               for key, value in obs.items()})
