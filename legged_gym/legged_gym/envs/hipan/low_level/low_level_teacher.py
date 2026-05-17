# legged_gym/envs/hipan/low_level/low_level_teacher.py
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.helpers import class_to_dict


class LowLevelTeacher(LeggedRobot):
    """HiPAN low-level teacher policy piL_T: PPO training, privileged info -> joint actions"""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_low_obs = 57
        self.num_command = 5

    def _init_buffers(self):
        super()._init_buffers()

        # Acquire rigid body state tensor (needed for foot_pos in observations)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.foot_pos = self.rigid_body_states[:, self.feet_indices, :3]

        # Override base commands to 5D: [vx, vy, wz, h, roll]
        self.commands = torch.zeros(
            self.num_envs, self.num_command,
            dtype=torch.float, device=self.device,
            requires_grad=False,
        )
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel,
             self.obs_scales.lin_vel,
             self.obs_scales.ang_vel,
             1.0, 1.0],
            device=self.device, requires_grad=False,
        )

        # second_last_actions for smooth_action reward (not in base class)
        self.second_last_actions = torch.zeros(
            self.num_envs, self.num_actions,
            dtype=torch.float, device=self.device,
            requires_grad=False,
        )

        # Domain parameter encoder: height_samples(187) + params(5) -> zd(32)
        self.domain_encoder = torch.nn.Sequential(
            torch.nn.Linear(187 + 5, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
        ).to(self.device)

        # Backbone network: op(57) + c(5) + xm(5) + zd(32) -> delta_q(12)
        input_dim = 57 + 5 + 5 + 32
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 12),  # delta_q: 12 joint offsets
        ).to(self.device)

    def post_physics_step(self):
        """Override to add rigid_body_state refresh and second_last_actions maintenance."""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        # Maintain action history for smooth_action reward
        self.second_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _post_physics_step_callback(self):
        """Override to avoid base heading_command logic that corrupts 5D commands."""
        env_ids = (
            self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0
        ).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if (self.cfg.domain_rand.push_robots
                and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0)):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """Sample 5D commands [vx, vy, wz, h, roll]."""
        self.commands[env_ids, 0] = (
            torch.rand(len(env_ids), device=self.device)
            * (self.command_ranges["lin_vel_x"][1] - self.command_ranges["lin_vel_x"][0])
            + self.command_ranges["lin_vel_x"][0]
        )
        self.commands[env_ids, 1] = (
            torch.rand(len(env_ids), device=self.device)
            * (self.command_ranges["lin_vel_y"][1] - self.command_ranges["lin_vel_y"][0])
            + self.command_ranges["lin_vel_y"][0]
        )
        self.commands[env_ids, 2] = (
            torch.rand(len(env_ids), device=self.device)
            * (self.command_ranges["ang_vel_yaw"][1] - self.command_ranges["ang_vel_yaw"][0])
            + self.command_ranges["ang_vel_yaw"][0]
        )
        self.commands[env_ids, 3] = (
            torch.rand(len(env_ids), device=self.device)
            * (self.command_ranges["height"][1] - self.command_ranges["height"][0])
            + self.command_ranges["height"][0]
        )
        self.commands[env_ids, 4] = (
            torch.rand(len(env_ids), device=self.device)
            * (self.command_ranges["roll"][1] - self.command_ranges["roll"][0])
            + self.command_ranges["roll"][0]
        )

    def _get_domain_params(self):
        """Build privileged domain parameter vector: height samples + friction/mass/motor."""
        if self.cfg.terrain.measure_heights:
            height_feat = self.measured_heights
        else:
            height_feat = torch.zeros(self.num_envs, 187, device=self.device)
        friction = (
            self.friction_coeffs.squeeze(-1)
            if hasattr(self, 'friction_coeffs')
            else torch.ones(self.num_envs, device=self.device)
        )
        mass = torch.ones(self.num_envs, device=self.device)
        motor_strength = getattr(
            self, 'motor_strength',
            torch.ones(2, self.num_envs, self.num_dof, device=self.device),
        )
        motor_mean = motor_strength.mean(dim=[0, 2])
        params = torch.stack(
            [friction, mass, motor_mean[0], motor_mean[1],
             torch.zeros(self.num_envs, device=self.device)], dim=1,
        )
        return torch.cat([height_feat, params], dim=1)

    def compute_observations(self):
        """Observation: op(57) + c(5). Teacher internally uses privileged xm + zd."""
        # op: proprioceptive (57-dim)
        self.proprio_buf = torch.cat((
            self.dof_pos - self.default_dof_pos.squeeze(0),          # 12
            self.dof_vel * self.obs_scales.dof_vel,                   # 12
            self.foot_pos.reshape(self.num_envs, -1),                  # 12
            (self.contact_forces[:, self.feet_indices, 2] > 1.).float(),  # 4
            self.projected_gravity[:, :2],                             # 2
            self.base_ang_vel * self.obs_scales.ang_vel,              # 3
            self.last_actions,                                          # 12
        ), dim=-1)  # 57-dim

        # Privileged motion state xm (5-dim)
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1,
        )
        body_roll = torch.atan2(
            self.projected_gravity[:, 0],
            self.projected_gravity[:, 2],
        ).unsqueeze(1)
        self.xm = torch.cat((
            self.base_lin_vel,   # v_B (3)
            base_height,          # h_B (1)
            body_roll,            # theta_x (1)
        ), dim=-1)  # 5-dim

        # Domain parameter latent vector zd (32-dim)
        domain_input = self._get_domain_params()
        self.zd = self.domain_encoder(domain_input)

        # Full privileged observation (used by PPO algorithm via self.obs_buf)
        self.obs_buf = torch.cat((
            self.proprio_buf,
            self.commands * self.commands_scale,
            self.xm,
            self.zd,
        ), dim=-1)

    def _compute_torques(self, actions):
        """PD control: actions are joint offsets delta_q from default pose."""
        self.actions = actions
        target_pos = self.default_dof_pos.squeeze(0) + actions * self.cfg.control.action_scale
        torques = self.p_gains * (target_pos - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _prepare_reward_function(self):
        """HiPAN low-level 12 reward terms."""
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name != "termination":
                name_lower = '_reward_' + name
                if hasattr(self, name_lower):
                    self.reward_names.append(name)
                    self.reward_functions.append(getattr(self, name_lower))
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float,
                              device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    # ---------- Reward Functions (HiPAN Table III) ----------
    def _reward_velocity_tracking(self):
        vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-vel_error / self.cfg.rewards.tracking_sigma_vel)

    def _reward_yaw_tracking(self):
        yaw_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-yaw_error / self.cfg.rewards.tracking_sigma_yaw)

    def _reward_height_tracking(self):
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        height_error = torch.abs(self.commands[:, 3] - base_height)
        return torch.exp(-height_error / self.cfg.rewards.tracking_sigma_height)

    def _reward_roll_tracking(self):
        body_roll = torch.atan2(
            self.projected_gravity[:, 0], self.projected_gravity[:, 2])
        roll_error = torch.abs(self.commands[:, 4] - body_roll)
        return torch.exp(-roll_error / self.cfg.rewards.tracking_sigma_roll)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_smooth_action(self):
        return torch.sum(torch.square(
            self.actions - 2.0 * self.last_actions + self.second_last_actions), dim=1)

    def _reward_body_orientation(self):
        return torch.abs(self.projected_gravity[:, 1])

    def _reward_body_velocity(self):
        return (torch.square(self.base_lin_vel[:, 2])
                + torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1))

    def _reward_smooth_joint_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_smooth_joint_acc(self):
        return torch.sum(torch.square(
            (self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_torque_usage(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_joint_limit(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return (torch.sum(out_of_limits, dim=1) > 0).float()

    def _reward_collision(self):
        return (torch.any(torch.norm(
            self.contact_forces[:, self.penalised_contact_indices, :],
            dim=-1) > 0.1, dim=1)).float()
