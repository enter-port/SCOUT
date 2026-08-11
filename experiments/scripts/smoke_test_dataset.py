"""Fail-fast: confirm the core hdf5 loads through LPB's RobomimicReplayImageDataset
with abs_action=True (reads 'abs_actions' 7-d -> converts to 10-d 6d). Catches the
abs_actions/KeyError risk before launching long trainings."""
import sys
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
shape_meta = {'obs': {
    'agentview_image': {'shape':[3,84,84],'type':'rgb'},
    'robot0_eye_in_hand_image': {'shape':[3,84,84],'type':'rgb'},
    'robot0_eef_pos':{'shape':[3]}, 'robot0_eef_quat':{'shape':[4]}, 'robot0_gripper_qpos':{'shape':[2]}},
    'action':{'shape':[10]}}
ds = RobomimicReplayImageDataset(
    shape_meta=shape_meta, dataset_path=sys.argv[1],
    horizon=16, pad_before=1, pad_after=7, n_obs_steps=2,
    abs_action=True, rotation_rep='rotation_6d', use_cache=False, val_ratio=0.02)
print("DATASET_OK len=", len(ds),
      "| action=", tuple(ds.replay_buffer['action'].shape),
      "| abs_action=", tuple(ds.replay_buffer['abs_action'].shape))
