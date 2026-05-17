# legged_gym/scripts/train_high_teacher.py
"""HiPAN high-level teacher PPO training script.

Workflow:
  1. Load a trained low-level teacher checkpoint as the frozen student.
  2. Create a HighLevelTeacher environment with its own WFC terrain sim.
  3. Inject the low-level env + policy into the high-level env via set_low_level().
  4. Run PPO training on the high-level policy.

Usage:
  python legged_gym/scripts/train_high_teacher.py --task=hipan_high_teacher [--headless] [--num_envs=1024]
"""

import os
import torch
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.hipan.pgcl import PGCLManager, AStarPlanner


def train_high_teacher(args):
    # ------------------------------------------------------------------
    # 1. Load trained low-level teacher (labeling oracle / frozen student)
    # ------------------------------------------------------------------
    print("Loading trained low-level teacher checkpoint ...")
    from legged_gym.scripts.train_low_student import load_low_teacher

    low_env, low_policy = load_low_teacher(args, num_envs=1024)
    print("Low-level teacher loaded successfully.")

    # ------------------------------------------------------------------
    # 2. Create high-level environment
    # ------------------------------------------------------------------
    print("Creating HighLevelTeacher environment ...")
    high_cfg, high_train_cfg = task_registry.get_cfgs(name="hipan_high_teacher")
    high_cfg.env.num_envs = 1024

    high_env, high_env_cfg = task_registry.make_env(
        name="hipan_high_teacher", args=args, env_cfg=high_cfg,
    )

    # ------------------------------------------------------------------
    # 3. Inject low-level env and policy into high-level teacher
    # ------------------------------------------------------------------
    high_env.set_low_level(low_env, low_policy)
    print("Low-level env/policy injected into high-level teacher.")

    # ------------------------------------------------------------------
    # 4. Setup PGCL managers (re-init with high-level config params)
    # ------------------------------------------------------------------
    planner = AStarPlanner()
    for i in range(high_env.num_envs):
        high_env.pgcl_managers[i] = PGCLManager(
            initial_d=high_cfg.nav.pgcl_initial_d,
            d_step=high_cfg.nav.pgcl_d_step,
            path_planner=planner,
        )
    print(f"PGCL managers initialized for {high_env.num_envs} environments.")

    # ------------------------------------------------------------------
    # 5. PPO training
    # ------------------------------------------------------------------
    print("Starting PPO training for high-level teacher ...")
    runner, train_cfg = task_registry.make_alg_runner(
        env=high_env, name="hipan_high_teacher", args=args, train_cfg=high_train_cfg,
    )
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )

    print("High-level teacher training completed.")


if __name__ == "__main__":
    args = get_args()
    args.task = "hipan_high_teacher"
    train_high_teacher(args)
