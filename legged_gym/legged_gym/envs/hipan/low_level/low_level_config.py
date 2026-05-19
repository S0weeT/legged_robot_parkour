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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class LowLevelCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 99   # op(57) + c(5) + xm(5) + zd(32)
        num_privileged_obs = None
        num_actions = 12
        episode_length_s = 20
        send_timeouts = True

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = True
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.25, 0.25, 0.25, 0.25, 0.0]  # slopes/rough/stairs_up/stairs_down
        measure_heights = True

    class commands(LeggedRobotCfg.commands):
        num_commands = 5   # [vx, vy, ωz, h, θx]
        resampling_time = 10.
        heading_command = False  # HiPAN does not use heading mode, uses ωz directly

        class ranges:
            lin_vel_x = [0.3, 0.5]   # Grid-Adaptive Curriculum initial range
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.2, 0.2]
            height = [0.28, 0.30]
            roll = [-0.2, 0.2]

        # Grid-Adaptive Curriculum parameters
        grid_adaptive = False
        ga_threshold = 0.8          # tracking accuracy threshold
        ga_expand_step = 0.1        # expansion step size
        ga_max_ranges = {
            'lin_vel_x': [-1.5, 1.5],
            'lin_vel_y': [-1.0, 1.0],
            'ang_vel_yaw': [-1.5, 1.5],
            'height': [0.1, 0.4],
            'roll': [-1.0, 1.0],
        }

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.30]
        default_joint_angles = {
            'FL_hip_joint': 0.1, 'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1, 'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8, 'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8, 'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5, 'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5, 'RR_calf_joint': -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 30}
        damping = {'joint': 0.75}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        flip_visual_attachments = True

    class domain_rand:
        randomize_friction = True
        friction_range = [0.7, 1.2]
        randomize_base_mass = True
        added_mass_range = [0.0, 3.0]
        randomize_base_com = True
        added_com_range = [-0.1, 0.1]
        randomize_motor = True
        motor_strength_range = [0.9, 1.1]
        push_robots = True
        push_interval_s = 10.0
        max_push_vel_xy = 1.0

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        tracking_sigma_vel = 0.1
        tracking_sigma_yaw = 0.25
        tracking_sigma_height = 0.025  # 0.0025*10 (pre-dt compensation)
        tracking_sigma_roll = 0.05

        # Energy regularization (Berkeley adaptive CoT)
        en_alpha = 0.5           # energy weight relative to motion tracking
        en_sigma = 100.0         # CoT normalization factor (W per m/s)
        en_eps = 0.05            # prevent division by zero at low speeds

        class scales:
            # -- R_motion: command tracking (dt-scaled, additive) --
            velocity_tracking = 3.0
            yaw_tracking = 0.4
            height_tracking = 1.5
            roll_tracking = 0.5

            # -- R_aux: stability gate (no dt scaling, in exp(-R_aux)) --
            body_orientation = 1.5
            body_velocity = 0.2
            collision = 1.0
            action_rate = 0.02
            smooth_action = 0.01
            smooth_joint_vel = 0.002
            smooth_joint_acc = 0.0000005
            torque_usage = 0.0005
            joint_limit = 0.5
            dof_pos = 0.05
            feet_air_time = 0.5

            # -- Closed --
            gait_phase = 0.0

        gait_frequency = 1.8
        gait_offsets = [0.0, 0.5, 0.5, 0.0]

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            dof_pos = 1.0
            dof_vel = 0.05
            ang_vel = 0.25
            lin_vel = 2.0
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        gravity = [0., 0., -9.81]

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1


class LowLevelCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'hipan_low_teacher'
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 6000

        save_interval = 50
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
