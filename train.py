"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
import torch
import torch.multiprocessing
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# The server's torch_shm_manager helper binary is unsupported there
# ("Operation not supported"), which breaks DataLoader worker -> main tensor
# transfer under the default shared-memory strategy. /dev/shm itself is large
# and healthy, so switch to the file_system strategy to make
# dataloader.num_workers > 0 usable (workers only; no effect at num_workers=0).
torch.multiprocessing.set_sharing_strategy('file_system')

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
