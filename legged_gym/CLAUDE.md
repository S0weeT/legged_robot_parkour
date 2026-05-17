# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Fork of ETH Zurich's `legged_gym` — GPU-accelerated reinforcement learning for legged robots using NVIDIA Isaac Gym + PPO (`rsl_rl`). Current branch (`goal`) trains a **Unitree Go2** to navigate a custom 9-section competition terrain by following sequential waypoints, using gait-phase rewards and goal-directed velocity projection.

## Common commands

```bash
# Train Go2 on competition terrain
python legged_gym/scripts/train.py --task=go2

# Train ANYmal on flat terrain
python legged_gym/scripts/train.py --task=anymal_c_flat

# Resume training from last checkpoint
python legged_gym/scripts/train.py --task=go2 --resume

# Run trained policy
python legged_gym/scripts/play.py --task=go2

# Smoke test (creates env, runs zero actions for 10 episodes)
python legged_gym/tests/test_env.py --task=go2
```

Key CLI args: `--headless`, `--num_envs=N`, `--seed=N`, `--max_iterations=N`, `--experiment_name=NAME`, `--run_name=NAME`.

## Architecture

**Inheritance chain:**
`BaseTask` ([base_task.py](envs/base/base_task.py)) — owns Isaac Gym sim, viewer, buffer allocation, reset/step interface
→ `LeggedRobot` ([legged_robot.py](envs/base/legged_robot.py)) — adds terrain creation, PD control, reward functions, domain randomization, observation computation
→ `Go2Robot` ([go2_robot.py](envs/go2/go2_robot.py)) — competition-specific overrides (waypoint navigation, gait phase, competition terrain, custom termination)

**Config system:**
[base_config.py](envs/base/base_config.py) auto-instantiates nested classes recursively on `__init__`. Every config attribute defined as a nested class becomes an instance. Configs follow the same inheritance chain as environments: `BaseConfig` → `LeggedRobotCfg` / `LeggedRobotCfgPPO` ([legged_robot_config.py](envs/base/legged_robot_config.py)) → `Go2RoughCfg` / `Go2RoughCfgPPO` ([go2_config.py](envs/go2/go2_config.py)).

**Task registry** ([task_registry.py](utils/task_registry.py)): maps task name string → (EnvClass, EnvConfig, TrainConfig). All tasks registered in [envs/__init__.py](envs/__init__.py). `make_env()` and `make_alg_runner()` handle config overrides from CLI.

**Training loop:** [train.py](scripts/train.py) calls `task_registry.make_env()` + `make_alg_runner()`, then `runner.learn()`. Uses PPO from `rsl_rl`.

**Reward mechanism:** Any non-zero scale in `cfg.rewards.scales` causes the corresponding `_reward_<NAME>()` method to be called each step. Zero-scale rewards are stripped at init (no-op).

## Key modifications on this branch (vs upstream)

- **Waypoint system**: `cfg.env.target_waypoints` list of `[x, y, z]` waypoints. `current_waypoint_idx` tracks each env's progress; `current_target_pos` is the active waypoint. Reached when within `cfg.env.waypoint_threshold` distance. Reset on episode end.
- **`_reward_tracking_goal_vel`**: projects world-frame XY velocity onto the direction toward the current waypoint, clamped at max speed.
- **`_reward_gait_phase`**: cosine-based continuous contact matching against desired trot gait (offsets `[0, 0.5, 0.5, 0]`) with Hz from `cfg.rewards.gait_frequency`. Uses foot contact forces normalized to [0,1] rather than binary threshold.
- **Command hijacking**: `post_physics_step` overwrites `self.commands[:, 0:2]` with local-frame direction toward waypoint, so the policy's velocity tracking behavior steers toward the goal.
- **`_reward_lin_vel_y`**: penalizes lateral body-frame velocity to reduce sway.
- **Competition terrain**: `mesh_type='competition'` triggers `create_competition_map()` which builds a 9-row custom course (flat → pyramid slope → random uniform → discrete obstacles → waves → stairs up → stairs down → stepping stones → flat) as a single trimesh.
- **Custom termination**: `base_z < -3` fall detection.
- **`_reward_dof_pos`**: penalizes deviation from default standing pose.

## Adding a new environment

1. Create `<robot>_config.py` (inherit from `LeggedRobotCfg`/`LeggedRobotCfgPPO`) and `<robot>_robot.py` (inherit from `LeggedRobot`) in `envs/<name>/`
2. Override `class asset` (URDF path, foot names, contact bodies), `class init_state` (joint angles), `class control` (PD gains), `class rewards.scales` (enable/disable reward terms)
3. Override reward methods and/or `post_physics_step`, `check_termination`, etc. as needed
4. Register in [envs/__init__.py](envs/__init__.py)
5. Add URDF to `resources/robots/<name>/`

## Dependencies

`isaacgym` (Preview 3, GPU PhysX sim), `rsl_rl` (PPO), `pytorch`, `matplotlib`, `numpy`.
