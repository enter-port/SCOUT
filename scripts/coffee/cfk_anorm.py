"""Mean row-norm of the VIB action chunk on the SAME fixed batch the grad
probes use (torch.manual_seed(0), num_workers=0, feature_cache off, B=128).
Prints |a| for the R = eta*sqrt(1-abar)*|dKL/da| / |a| table."""
import argparse
import sys

sys.path.insert(0, ".")

import torch
import yaml
from easydict import EasyDict

from scout.train_vib import make_dataloader, _slice_transition
from dyn_model.datasets.img_transforms import get_eval_crop_transform_resnet

ap = argparse.ArgumentParser()
ap.add_argument("--vib-config", required=True)
ap.add_argument("--data-hdf5", required=True)
ap.add_argument("--task", required=True)
a = ap.parse_args()

cfg = yaml.safe_load(open(a.vib_config))
cfg["dataset"]["zarr_path"] = a.data_hdf5
cfg["dataset"]["num_workers"] = 0
cfg["dataset"]["feature_cache"] = False
cfg["batch_size"] = 128
torch.manual_seed(0)
loader, _ds = make_dataloader(EasyDict(cfg))
batch = next(iter(loader))
t_crop = get_eval_crop_transform_resnet(84, 76)
obs_t, a_t, _, _ = _slice_transition(batch, torch.device("cuda"), t_crop)
n = a_t.flatten(1).norm(dim=1)
print("NORM task=%s B=%d |a|mean=%.4f |a|median=%.4f std(a)=%.5f a_dim=%d" % (
    a.task, int(n.shape[0]), float(n.mean()), float(n.median()),
    float(a_t.std(dim=0).mean()), int(a_t.shape[1])))
