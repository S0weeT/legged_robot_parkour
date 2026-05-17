# legged_gym/envs/hipan/high_level/high_level_teacher.py
"""HiPAN high-level teacher policy piH_T: dual-map perception + PPO -> 5D commands.

Observation: M_3D_enc(16) + M_2.5D_enc(16) + proprio(62) + subgoal_body(3) = 97-dim
Action:       [vx, vy, wz, h, roll]  (5D command fed to frozen low-level student)

Architecture:
  - Inherits from BaseTask (NOT LeggedRobot) — owns a separate Isaac Gym sim
    that hosts only the WFC terrain mesh (no robot assets).
  - Embeds a frozen low-level student: high-level PPO produces a 5D command c,
    low-level runs 5 steps at 50Hz (100 ms) with fixed c, then high-level reads
    the resulting robot state and computes reward.
  - Uses PGCL (Path-Guided Curriculum Learning) to progressively extend
    navigation horizons by removing intermediate subgoals.
  - 10 reward terms following HiPAN Table II.
"""

import torch
import numpy as np
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.hipan.terrain.wfc_terrain import WFCTerrain
from legged_gym.envs.hipan.pgcl import PGCLManager, AStarPlanner, StateCountExploration
from legged_gym.utils.helpers import class_to_dict


class HighLevelTeacher(BaseTask):
    """HiPAN high-level teacher: dual-map + PPO -> 5D commands, embedding frozen low-level student."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        self.headless = headless
        self.low_level_env = None
        self.low_level_policy = None

        # Parse command ranges and observation scales before base init
        self._parse_high_cfg(cfg)

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Post-init: allocate extra buffers and prepare reward scales
        self._init_buffers()
        self._prepare_reward_function()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _parse_high_cfg(self, cfg):
        """Extract command ranges and scaling parameters from config."""
        cr = cfg.commands.ranges
        self.command_ranges = {
            "lin_vel_x": list(cr.lin_vel_x),
            "lin_vel_y": list(cr.lin_vel_y),
            "ang_vel_yaw": list(cr.ang_vel_yaw),
            "height": list(cr.height),
            "roll": list(cr.roll),
        }

        # Observation scales for proprioceptive components
        obs_sc = cfg.normalization.obs_scales
        self.obs_scales = {
            "dof_pos": getattr(obs_sc, "dof_pos", 1.0),
            "dof_vel": getattr(obs_sc, "dof_vel", 0.05),
            "ang_vel": getattr(obs_sc, "ang_vel", 0.25),
            "lin_vel": getattr(obs_sc, "lin_vel", 2.0),
            "height_measurements": getattr(obs_sc, "height_measurements", 5.0),
        }

    # ------------------------------------------------------------------
    # Isaac Gym sim (terrain only, no robots)
    # ------------------------------------------------------------------

    def set_low_level(self, low_level_env, low_level_policy):
        """Inject trained low-level student env and policy after construction.

        Args:
            low_level_env: LowLevelStudent (or LowLevelTeacher) environment
                           with its own Isaac Gym sim and robot assets.
            low_level_policy: Callable (obs) -> actions_mean (deterministic
                              PPO actor or DAgger student forward).
        """
        self.low_level_env = low_level_env
        self.low_level_policy = low_level_policy

    def create_sim(self):
        """Create sim with WFC terrain meshes (no robots — those live in the low-level sim)."""
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id,
            self.physics_engine, self.sim_params,
        )
        self._create_ground_plane()
        self._create_wfc_envs()

    def _create_ground_plane(self):
        """Flat ground plane underneath the WFC terrain."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_wfc_envs(self):
        """Create WFC terrain trimesh and load into sim."""
        wfc = WFCTerrain(self.cfg.terrain)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = wfc.vertices.shape[0]
        tm_params.nb_triangles = wfc.triangles.shape[0]
        tm_params.transform.p.x = -wfc.tot_cols * self.cfg.terrain.horizontal_scale / 2
        tm_params.transform.p.y = -wfc.tot_rows * self.cfg.terrain.horizontal_scale / 2
        self.gym.add_triangle_mesh(
            self.sim,
            wfc.vertices.flatten().astype(np.float32),
            wfc.triangles.flatten().astype(np.int32),
            tm_params,
        )
        self.wfc_terrain = wfc
        self._hf_tensor = torch.from_numpy(
            wfc.heightsamples.astype(np.float32)
        ).to(self.sim_device)

    # ------------------------------------------------------------------
    # Buffers and networks
    # ------------------------------------------------------------------

    def _init_buffers(self):
        """Allocate command buffers, PGCL managers, map encoders, and backbone."""
        n = self.num_envs

        # Command buffers
        self.commands = torch.zeros(n, 5, dtype=torch.float, device=self.device)
        self.last_command = torch.zeros_like(self.commands)
        self.second_last_command = torch.zeros_like(self.commands)

        # PGCL per environment
        planner = AStarPlanner()
        self.pgcl_managers = [
            PGCLManager(
                initial_d=self.cfg.nav.pgcl_initial_d,
                d_step=self.cfg.nav.pgcl_d_step,
                path_planner=planner,
            )
            for _ in range(n)
        ]
        self.state_explorer = StateCountExploration(resolution=0.1)

        # Start and goal positions per env
        self.global_goal = torch.zeros(n, 3, device=self.device)
        self.current_subgoal = torch.zeros(n, 3, device=self.device)
        self.start_positions = torch.zeros(n, 3, device=self.device)

        # ---- 3D map encoder (voxel occupancy -> 16-dim feature) ----
        self.map3d_encoder = torch.nn.Sequential(
            torch.nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 16),
        ).to(self.device)

        # ---- 2.5D map encoder (elevation map -> 16-dim feature) ----
        self.map2d_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 16),
        ).to(self.device)

        # Map encoders are used as fixed feature extractors during PPO training
        # (they produce the observation that PPO learns on).
        for param in self.map3d_encoder.parameters():
            param.requires_grad = False
        for param in self.map2d_encoder.parameters():
            param.requires_grad = False

        # ---- Backbone (placed here for architectural reference; PPO creates its own actor-critic) ----
        backbone_input_dim = 16 + 16 + 62 + 3  # map_feat(32) + proprio(62) + subgoal_body(3)
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(backbone_input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 5),
        ).to(self.device)

        # Re-calculate actual observation dimension and pad if needed
        self._actual_obs_dim = backbone_input_dim  # 97
        if self._actual_obs_dim < self.num_obs:
            self._obs_pad_dim = self.num_obs - self._actual_obs_dim
        else:
            self._obs_pad_dim = 0

    # ------------------------------------------------------------------
    # High-level step (bridge to low-level)
    # ------------------------------------------------------------------

    def step(self, actions):
        """High-level step: output 5D command c -> run low-level for N steps -> compute reward.

        Returns:
            obs_buf, privileged_obs (None), rew_buf, reset_buf, extras
        """
        if self.low_level_env is None:
            # Called before set_low_level() (e.g. during on-policy runner init).
            # Return dummy data so the runner can determine shapes.
            self.obs_buf = torch.zeros(
                self.num_envs, self.num_obs, dtype=torch.float, device=self.device,
            )
            self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.extras = {}
            return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

        # 1. Scale actions from [-1, 1] to physical command ranges
        c = self._scale_actions(actions)

        # 2. Embed low-level: run N steps at 50 Hz with fixed command
        low_env = self.low_level_env
        low_policy = self.low_level_policy
        n_steps = self.cfg.nav.low_level_steps_per_high

        # Use at most low_env.num_envs environments
        n_low = min(self.num_envs, low_env.num_envs)

        for _ in range(n_steps):
            # Re-set commands each low-level step to prevent
            # _resample_commands() from overwriting our values.
            low_env.commands[:n_low] = c[:n_low]

            low_obs = low_env.get_observations()
            if low_obs.dim() == 1:
                low_obs = low_obs.unsqueeze(0)
            low_obs = low_obs[:n_low]

            with torch.no_grad():
                low_actions = low_policy(low_obs)

            low_env.step(low_actions)

        # 3. Propagate low-level terminations to high-level reset buffer
        self.reset_buf[:n_low] = low_env.reset_buf[:n_low]
        self.reset_buf[n_low:] = 1  # any excess envs reset immediately

        # 4. Update episode length and timeout tracking
        self.episode_length_buf += 1
        max_ep_len = int(self.cfg.env.episode_length_s * self.cfg.nav.high_level_freq)
        self.time_out_buf[:] = self.episode_length_buf >= max_ep_len
        self.reset_buf[self.time_out_buf] = 1

        # 5. Save commands for reward computation
        self.commands[:] = actions

        # 6. Compute observations and reward
        self.compute_observations()
        self.compute_reward()

        # 7. Clip observations
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def _scale_actions(self, actions):
        """Scale normalized actions [-1, 1] to physical command ranges using config."""
        c = torch.zeros_like(actions)
        cr = self.command_ranges

        # Map each dim from [-1, 1] to [min, max]
        c[:, 0] = (actions[:, 0] + 1.0) / 2.0 * (cr["lin_vel_x"][1] - cr["lin_vel_x"][0]) + cr["lin_vel_x"][0]
        c[:, 1] = (actions[:, 1] + 1.0) / 2.0 * (cr["lin_vel_y"][1] - cr["lin_vel_y"][0]) + cr["lin_vel_y"][0]
        c[:, 2] = (actions[:, 2] + 1.0) / 2.0 * (cr["ang_vel_yaw"][1] - cr["ang_vel_yaw"][0]) + cr["ang_vel_yaw"][0]
        c[:, 3] = (actions[:, 3] + 1.0) / 2.0 * (cr["height"][1] - cr["height"][0]) + cr["height"][0]
        c[:, 4] = (actions[:, 4] + 1.0) / 2.0 * (cr["roll"][1] - cr["roll"][0]) + cr["roll"][0]

        return c

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def compute_observations(self):
        """Build high-level observation: map encoding + proprio + subgoal.

        obs_buf = [M_3D_enc(16), M_2.5D_enc(16), proprio(62), subgoal_body(3)] = 97-dim
        Padded to self.num_obs (256) with zeros if needed.
        """
        low_env = self.low_level_env
        n = min(self.num_envs, low_env.num_envs)

        # ---- Proprioceptive state from low-level ----
        # Replicate low-level teacher's proprio_buf computation
        op = torch.cat((
            low_env.dof_pos[:n] - low_env.default_dof_pos.squeeze(0),          # 12
            low_env.dof_vel[:n] * self.obs_scales["dof_vel"],                   # 12
            low_env.foot_pos[:n].reshape(n, -1),                                 # 12
            (low_env.contact_forces[:n, low_env.feet_indices, 2] > 1.).float(),  # 4
            low_env.projected_gravity[:n, :2],                                    # 2
            low_env.base_ang_vel[:n] * self.obs_scales["ang_vel"],              # 3
            low_env.last_actions[:n],                                             # 12
        ), dim=-1)  # 57-dim

        hl_prop = torch.cat([op, self.commands[:n]], dim=1)  # 57 + 5 = 62

        # ---- Relative subgoal position in body frame ----
        robot_pos = low_env.root_states[:n, :3]
        robot_quat = low_env.base_quat[:n]
        delta_world = self.current_subgoal[:n] - robot_pos
        delta_body = quat_rotate_inverse(robot_quat, delta_world)

        # ---- Build map tensors from WFC terrain ----
        M3D, M2D = self._get_privileged_maps(robot_pos)

        with torch.no_grad():
            z_m3d = self.map3d_encoder(M3D)  # (n, 16)
            z_m2d = self.map2d_encoder(M2D)  # (n, 16)
        map_code = torch.cat([z_m3d, z_m2d], dim=1)  # 32-dim

        # ---- Concatenate full observation ----
        obs = torch.cat([map_code, hl_prop, delta_body], dim=1)  # 97-dim

        # Pad to match self.num_obs if the config expects a larger buffer
        if self._obs_pad_dim > 0:
            padding = torch.zeros(obs.shape[0], self._obs_pad_dim, device=self.device)
            obs = torch.cat([obs, padding], dim=1)

        # Handle extra envs beyond low_env's count
        if n < self.num_envs:
            full_obs = torch.zeros(self.num_envs, self.num_obs, device=self.device)
            full_obs[:n] = obs
            obs = full_obs

        self.obs_buf = obs
        return self.obs_buf

    def _get_privileged_maps(self, robot_positions):
        """Extract local M_3D and M_2.5D around each robot from WFC terrain (vectorized).

        Args:
            robot_positions: (N, 3) world-frame robot positions.

        Returns:
            M3D: (N, 1, D, H, W) 3D voxel occupancy grids.
            M2D: (N, 1, H, W) 2.5D elevation map crops.
        """
        n = robot_positions.shape[0]
        M3D_size = self.cfg.nav.map_3d_size
        M2D_size = self.cfg.nav.map_2d5_size

        if n == 0:
            return (
                torch.zeros(0, 1, *M3D_size, device=self.device),
                torch.zeros(0, 1, *M2D_size, device=self.device),
            )

        hf_tensor = self._hf_tensor
        hf_rows, hf_cols = hf_tensor.shape
        hs = self.cfg.terrain.horizontal_scale
        vs = self.cfg.terrain.vertical_scale
        wfc = self.wfc_terrain

        # World XY → heightfield indices for all robots at once
        cx = robot_positions[:, 0] / hs + wfc.tot_cols / 2.0  # (N,)
        cy = robot_positions[:, 1] / hs + wfc.tot_rows / 2.0  # (N,)
        rz = robot_positions[:, 2]                             # (N,)

        # ---- 2.5D map ----
        res_2d = self.cfg.nav.map_2d5_resolution
        half_w_2d = M2D_size[1] // 2
        half_h_2d = M2D_size[0] // 2

        dx_hf = (torch.arange(M2D_size[1], device=self.device) - half_w_2d).float() * (res_2d / hs)
        dy_hf = (torch.arange(M2D_size[0], device=self.device) - half_h_2d).float() * (res_2d / hs)

        ix = cx[:, None, None] + dx_hf[None, None, :]   # (N, 1, W)  → column indices
        iy = cy[:, None, None] + dy_hf[None, :, None]   # (N, H, 1)  → row indices

        valid_2d = (ix >= 0) & (ix < hf_cols) & (iy >= 0) & (iy < hf_rows)
        ix = ix.round().long().clamp(0, hf_cols - 1)
        iy = iy.round().long().clamp(0, hf_rows - 1)

        batch_M2D = hf_tensor[iy, ix] * vs      # (N, H, W)
        batch_M2D[~valid_2d] = 0.0
        batch_M2D = batch_M2D.unsqueeze(1)       # (N, 1, H, W)

        # ---- 3D map ----
        res_3d = self.cfg.nav.map_3d_resolution
        half_w_3d = M3D_size[2] // 2
        half_d_3d = M3D_size[1] // 2

        dx_3d = (torch.arange(M3D_size[2], device=self.device) - half_w_3d).float() * (res_3d / hs)
        dy_3d = (torch.arange(M3D_size[1], device=self.device) - half_d_3d).float() * (res_3d / hs)

        ix_3d = cx[:, None, None] + dx_3d[None, None, :]   # (N, 1, W)
        iy_3d = cy[:, None, None] + dy_3d[None, :, None]   # (N, H, 1)

        valid_3d = (ix_3d >= 0) & (ix_3d < hf_cols) & (iy_3d >= 0) & (iy_3d < hf_rows)
        ix_3d = ix_3d.round().long().clamp(0, hf_cols - 1)
        iy_3d = iy_3d.round().long().clamp(0, hf_rows - 1)

        terrain_heights = hf_tensor[iy_3d, ix_3d] * vs   # (N, H, W)
        terrain_heights[~valid_3d] = 0.0

        # Voxel occupancy: terrain above each Z level
        dz_offsets = (torch.arange(M3D_size[0], device=self.device) - M3D_size[0] // 2).float() * res_3d
        world_z = rz[:, None] + dz_offsets[None, :]       # (N, D)

        # (N, D, H, W): terrain_h > world_z for each depth slice
        occupied = terrain_heights[:, None, :, :] > world_z[:, :, None, None]
        batch_M3D = occupied.float().unsqueeze(1)          # (N, 1, D, H, W)

        return batch_M3D, batch_M2D

    # ------------------------------------------------------------------
    # Rewards (10 terms, HiPAN Table II)
    # ------------------------------------------------------------------

    def compute_reward(self):
        """10 reward terms per HiPAN Table II.

        All scales are accessed via self.reward_scales populated by
        _prepare_reward_function().
        """
        low_env = self.low_level_env
        n = min(self.num_envs, low_env.num_envs)

        robot_pos = low_env.root_states[:n, :2]
        dist_to_goal = torch.norm(self.current_subgoal[:n, :2] - robot_pos, dim=1)
        goal_reached = dist_to_goal < self.cfg.nav.goal_arrival_threshold

        # 1. Goal arrival
        r_goal = goal_reached.float() * self.reward_scales.get("goal_arrival", 0.0)

        # 2. Intrinsic Reward (state-count exploration bonus)
        ir_vals = torch.tensor([
            self.state_explorer.get_intrinsic_reward(pos.cpu().numpy())
            for pos in robot_pos
        ], device=self.device)
        r_ir = ir_vals * self.reward_scales.get("state_count", 0.0)

        # Update state counts
        for pos in robot_pos.cpu().numpy():
            self.state_explorer.increment(pos)

        # 3. Desired speed
        v_des = torch.rand(n, device=self.device) * (
            self.cfg.nav.desired_speed_range[1] - self.cfg.nav.desired_speed_range[0]
        ) + self.cfg.nav.desired_speed_range[0]
        r_speed = torch.exp(-torch.abs(
            v_des - torch.norm(low_env.base_lin_vel[:n, :2], dim=1)
        ) / self.cfg.rewards.tracking_sigma) * self.reward_scales.get("desired_speed", 0.0)

        # 4. Command rate (penalize large changes between consecutive commands)
        r_cmd_rate = torch.sum(torch.square(
            self.commands[:n] - self.last_command[:n]), dim=1
        ) * self.reward_scales.get("command_rate", 0.0)

        # 5. Smooth command (penalize second-order changes)
        r_cmd_smooth = torch.sum(torch.square(
            self.commands[:n] - 2.0 * self.last_command[:n] + self.second_last_command[:n]
        ), dim=1) * self.reward_scales.get("smooth_command", 0.0)

        # 6. Tracking error
        r_track = torch.norm(
            self.commands[:n, :2] - low_env.base_lin_vel[:n, :2], dim=1
        ) * self.reward_scales.get("tracking_error", 0.0)

        # 7. Body velocity (penalize z linear + xy angular velocity)
        r_body_vel = (
            torch.square(low_env.base_lin_vel[:n, 2])
            + torch.sum(torch.square(low_env.base_ang_vel[:n, :2]), dim=1)
        ) * self.reward_scales.get("body_velocity", 0.0)

        # 8. Nominal posture (penalize deviation from default joint angles)
        r_posture = torch.sum(torch.square(
            low_env.dof_pos[:n] - low_env.default_dof_pos[:n]
        ), dim=1) * self.reward_scales.get("nominal_posture", 0.0)

        # 9. Command limit (penalize total command magnitude exceeding threshold)
        r_cmd_limit = (
            torch.abs(self.commands[:n]).sum(dim=1) > 5.0
        ).float() * self.reward_scales.get("command_limit", 0.0)

        # 10. Collision (penalize body-part contacts)
        collision_mask = torch.any(torch.norm(
            low_env.contact_forces[:n, low_env.penalised_contact_indices, :], dim=-1
        ) > 0.1, dim=1)
        r_collision = collision_mask.float() * self.reward_scales.get("collision", 0.0)

        # Sum all reward terms
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.rew_buf[:n] = (
            r_goal + r_ir + r_speed + r_cmd_rate + r_cmd_smooth
            + r_track + r_body_vel + r_posture + r_cmd_limit + r_collision
        )

        # Update command history for next step
        self.second_last_command[:] = self.last_command[:]
        self.last_command[:] = self.commands[:]

    def _prepare_reward_function(self):
        """Populate self.reward_scales dict from config, filtering zero scales."""
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        for key in list(self.reward_scales.keys()):
            if self.reward_scales[key] == 0:
                self.reward_scales.pop(key)

    # ------------------------------------------------------------------
    # Resets
    # ------------------------------------------------------------------

    def reset_idx(self, env_ids):
        """Reset selected environments."""
        if len(env_ids) == 0:
            return

        self.commands[env_ids] = 0.
        self.last_command[env_ids] = 0.
        self.second_last_command[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.time_out_buf[env_ids] = False

        # Reset PGCL state for these envs (back to level 0)
        for i in env_ids.cpu().numpy():
            if i < len(self.pgcl_managers):
                self.pgcl_managers[i].current_d = self.cfg.nav.pgcl_initial_d
                self.pgcl_managers[i].level = 0

    def reset(self):
        """Reset all environments."""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device)
        )
        return obs, None

    # ------------------------------------------------------------------
    # Public interface (used by PPO runner)
    # ------------------------------------------------------------------

    def get_observations(self):
        return self.obs_buf
