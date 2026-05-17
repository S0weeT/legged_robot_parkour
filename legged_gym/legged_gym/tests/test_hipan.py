"""Smoke test: verify HiPAN low-level teacher environment initializes and steps."""
import sys
# Remove conflicting legged_gym installation from unitree_rl_gym
sys.path = [p for p in sys.path if 'unitree_rl_gym' not in p]
from isaacgym import gymapi  # must import before torch (Isaac Gym requirement)
import torch
from legged_gym.envs import *  # must come first to break circular import
from legged_gym.utils import get_args, task_registry


def test_low_teacher_env():
    """Basic environment creation and stepping test for low-level teacher."""
    args = get_args()
    args.task = "hipan_low_teacher"
    args.headless = True
    args.num_envs = 4

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 4
    env_cfg.terrain.mesh_type = 'plane'  # flat ground for quick testing
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Check observation shape
    obs = env.get_observations()
    assert obs.shape == (4, env_cfg.env.num_observations), \
        f"Obs shape mismatch: {obs.shape} vs {(4, env_cfg.env.num_observations)}"

    # Step the environment
    for _ in range(100):
        actions = torch.randn(4, env_cfg.env.num_actions, device=env.device)
        obs, priv, rew, done, info = env.step(actions)
        assert obs.shape == (4, env_cfg.env.num_observations)
        assert rew.shape == (4,)

    print("PASS: Low-level teacher environment works")
    return True


if __name__ == '__main__':
    test_low_teacher_env()
