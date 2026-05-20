# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def _print_action_stats(action_history, dof_pos_history, dof_target_history, cmd_history, actor_critic, obs, env):
    act_stack = torch.stack(action_history)          # [50, 12]
    target_stack = torch.stack(dof_target_history)   # [50, 12]
    pos_stack = torch.stack(dof_pos_history)         # [50, 12]
    cmd_stack = torch.stack(cmd_history)             # [50, 5]
    joint_names = ['FL_hip', 'FL_thigh', 'FL_calf', 'FR_hip', 'FR_thigh', 'FR_calf',
                   'RL_hip', 'RL_thigh', 'RL_calf', 'RR_hip', 'RR_thigh', 'RR_calf']

    # ——— Commands ———
    print("\n=== Command diagnostics (first 50 steps, robot 0) ===")
    print(f"  cmd_vx:  mean={cmd_stack[:, 0].mean().item():.4f}  std={cmd_stack[:, 0].std().item():.4f}  "
          f"min={cmd_stack[:, 0].min().item():.4f}  max={cmd_stack[:, 0].max().item():.4f}")
    print(f"  cmd_vy:  mean={cmd_stack[:, 1].mean().item():.4f}  std={cmd_stack[:, 1].std().item():.4f}  "
          f"min={cmd_stack[:, 1].min().item():.4f}  max={cmd_stack[:, 1].max().item():.4f}")
    print(f"  cmd_yaw: mean={cmd_stack[:, 2].mean().item():.4f}  std={cmd_stack[:, 2].std().item():.4f}  "
          f"min={cmd_stack[:, 2].min().item():.4f}  max={cmd_stack[:, 2].max().item():.4f}")
    print(f"  cmd_h:   mean={cmd_stack[:, 3].mean().item():.4f}  std={cmd_stack[:, 3].std().item():.4f}  "
          f"min={cmd_stack[:, 3].min().item():.4f}  max={cmd_stack[:, 3].max().item():.4f}")
    print(f"  cmd_roll:mean={cmd_stack[:, 4].mean().item():.4f}  std={cmd_stack[:, 4].std().item():.4f}  "
          f"min={cmd_stack[:, 4].min().item():.4f}  max={cmd_stack[:, 4].max().item():.4f}")

    # ——— Deterministic (inference) stats ———
    print("\n=== Action & PD Target diagnostics (first 50 steps, robot 0) ===")
    print(f"{'Joint':<12} {'Det mean':>10} {'Det std':>10} {'PD target':>10} {'Dof pos':>10} {'Default':>10} {'Track err':>10}")
    for j in range(12):
        print(f"{joint_names[j]:<12} {act_stack[:, j].mean().item():>10.4f} {act_stack[:, j].std().item():>10.4f} "
              f"{target_stack[:, j].mean().item():>10.4f} {pos_stack[:, j].mean().item():>10.4f} "
              f"{env.default_dof_pos[0, j].item():>10.4f} "
              f"{abs(target_stack[:, j].mean().item() - pos_stack[:, j].mean().item()):>10.4f}")
    print(f"\nDeterministic action global — mean: {act_stack.mean().item():.4f}  std: {act_stack.std().item():.4f}  "
          f"min: {act_stack.min().item():.4f}  max: {act_stack.max().item():.4f}")

    # ——— Stochastic (training) stats: sample 50x from current distribution ———
    with torch.no_grad():
        actor_critic.update_distribution(obs)  # sets self.distribution = Normal(mean, std)
        sto_samples = torch.stack([actor_critic.distribution.sample() for _ in range(50)])  # [50, N_envs, 12]
        sto_robot0 = sto_samples[:, 0, :]  # [50, 12]

    print(f"\n=== Stochastic (training) action stats: 50 samples from SAME obs (step 50) ===")
    print(f"  Policy learned std (per joint): {actor_critic.std.data}")
    print(f"  Policy std mean: {actor_critic.std.mean().item():.4f}")
    print(f"{'Joint':<12} {'Sto mean':>10} {'Sto std':>10} {'Det mean':>10} {'|mean diff|':>12}")
    for j in range(12):
        det_mean = act_stack[:, j].mean().item()
        sto_mean = sto_robot0[:, j].mean().item()
        sto_std = sto_robot0[:, j].std().item()
        print(f"{joint_names[j]:<12} {sto_mean:>10.4f} {sto_std:>10.4f} {det_mean:>10.4f} {abs(sto_mean - det_mean):>12.4f}")
    print(f"\nStochastic action global — mean: {sto_robot0.mean().item():.4f}  std: {sto_robot0.std().item():.4f}  "
          f"min: {sto_robot0.min().item():.4f}  max: {sto_robot0.max().item():.4f}")


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Print which checkpoint will be loaded
    load_run = train_cfg.runner.load_run if train_cfg.runner.load_run != -1 else '(latest)'
    checkpoint = train_cfg.runner.checkpoint if train_cfg.runner.checkpoint != -1 else '(latest)'
    print(f"\n{'='*60}")
    print(f"  Experiment: {train_cfg.runner.experiment_name}")
    print(f"  Run name:   {train_cfg.runner.run_name or '(latest run)'}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"{'='*60}\n")
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    # ——— Control experiment: keep training-like domain conditions ———
    # env_cfg.terrain.curriculum = False
    # env_cfg.terrain.mesh_type = 'plane'
    # env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy (triggers env.reset() → compute_observations())
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # NOTE: get_observations() must be AFTER make_alg_runner because
    # make_alg_runner triggers env.reset() which reassigns self.obs_buf
    obs = env.get_observations()

    # ——— Diagnostic: check domain_encoder zd statistics during play ———
    actor_critic = ppo_runner.alg.actor_critic
    with torch.no_grad():
        body = obs[:, :actor_critic.BODY_OBS_DIM]
        domain_params = obs[:, actor_critic.BODY_OBS_DIM:]
        zd = actor_critic.domain_encoder(domain_params)
        print("=== Domain Encoder zd statistics (play) ===")
        print(f"  zd mean: {zd.mean().item():.4f}")
        print(f"  zd std:  {zd.std().item():.4f}")
        print(f"  zd min:  {zd.min().item():.4f}")
        print(f"  zd max:  {zd.max().item():.4f}")
        print(f"  domain_params mean: {domain_params.mean().item():.4f}")
        print(f"  domain_params std:  {domain_params.std().item():.4f}")
        print(f"  domain_encoder fc0.weight norm: {actor_critic.domain_encoder[0].weight.norm().item():.4f}")
        print(f"  domain_encoder fc2.weight norm: {actor_critic.domain_encoder[2].weight.norm().item():.4f}")
        print(f"  domain_encoder fc4.weight norm: {actor_critic.domain_encoder[4].weight.norm().item():.4f}")
        # Check if actor uses zd: compare weight norms of body vs zd connections
        actor_fc0 = actor_critic.actor[0]  # nn.Linear(99, hidden)
        w_body = actor_fc0.weight[:, :actor_critic.BODY_OBS_DIM]   # connections from body (67 dims)
        w_zd = actor_fc0.weight[:, actor_critic.BODY_OBS_DIM:]     # connections from zd (32 dims)
        print(f"  actor fc0 weight[:, :67] (body→hidden) norm: {w_body.norm().item():.4f}")
        print(f"  actor fc0 weight[:, 67:] (zd→hidden)   norm: {w_zd.norm().item():.4f}")
        print(f"  zd/body norm ratio: {w_zd.norm().item() / w_body.norm().item():.4f}")
        print("==============================================")

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    #  get input 
    #vel =x
    #obs[:,12:15] = vel.
    # ——— Diagnostic: collect first N steps of actions ———
    action_history = []
    dof_pos_history = []
    dof_target_history = []
    cmd_history = []
    base_z_history = []
    base_tilt_history = []  # |gravity_xy| — 0 when upright, 1 when fully tilted
    feet_air_time_history = []
    gait_phase_history = []

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        if i < 50:
            action_history.append(actions[robot_index].clone())
            dof_pos_history.append(env.dof_pos[robot_index].clone())
            dof_target_history.append((env.default_dof_pos[robot_index] + actions[robot_index] * env.cfg.control.action_scale).clone())
            cmd_history.append(env.commands[robot_index, :5].clone())
            base_z_history.append(env.root_states[robot_index, 2].item())
            base_tilt_history.append(torch.norm(env.projected_gravity[robot_index, :2]).item())
            if hasattr(env, 'raw_rew'):
                feet_air_time_history.append(env.raw_rew.get('feet_air_time', torch.zeros(1))[robot_index].item())
                gait_phase_history.append(env.raw_rew.get('gait_phase', torch.zeros(1))[robot_index].item())
        elif i == 50:
            _print_action_stats(action_history, dof_pos_history, dof_target_history, cmd_history, actor_critic, obs, env)
            # Print base trajectory
            print("\n=== Base trajectory (first 50 steps, robot 0) ===")
            print(f"  base_z:  start={base_z_history[0]:.4f}  end={base_z_history[-1]:.4f}  min={min(base_z_history):.4f}  max={max(base_z_history):.4f}")
            print(f"  tilt:    start={base_tilt_history[0]:.4f}  end={base_tilt_history[-1]:.4f}  min={min(base_tilt_history):.4f}  max={max(base_tilt_history):.4f}")
            print(f"  (tilt = |gravity_xy|: 0=upright, 1=fully tilted)")
            if feet_air_time_history:
                fa = feet_air_time_history
                gp = gait_phase_history
                print(f"\n=== Raw gait reward values (first 50 steps, robot 0) ===")
                fa_scale = env.cfg.rewards.scales.feet_air_time
                gp_scale = env.cfg.rewards.scales.gait_phase
                print(f"  feet_air_time raw: mean={np.mean(fa):.6f}  std={np.std(fa):.6f}  min={np.min(fa):.6f}  max={np.max(fa):.6f}")
                print(f"  gait_phase raw:    mean={np.mean(gp):.4f}  std={np.std(gp):.4f}  min={np.min(gp):.4f}  max={np.max(gp):.4f}")
                print(f"  (feet_air_time × {fa_scale} × dt = {np.mean(fa)*fa_scale*0.005:.6f} /step)")
                print(f"  (gait_phase × {gp_scale} × dt  = {np.mean(gp)*gp_scale*0.005:.6f} /step)")
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
