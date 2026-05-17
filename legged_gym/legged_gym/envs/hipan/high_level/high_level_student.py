# legged_gym/envs/hipan/high_level/high_level_student.py
"""HiPAN high-level student policy: depth images + GRU -> scene latent -> 5D commands.

Replaces the teacher's privileged M_3D / M_2.5D map encoders with a domain-randomized
depth-image pipeline (CNN + GRU) that produces an equivalent z_s scene latent.
Trained via DAgger online distillation from a frozen HighLevelTeacher.
"""

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.hipan.high_level.high_level_teacher import HighLevelTeacher


class HighLevelStudent(HighLevelTeacher):
    """HiPAN high-level student: depth images + GRU -> 5D commands, distilled from teacher."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gru_hidden = None
        self.depth_h = 180
        self.depth_w = 320
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()

        # Depth image encoder: CNN feature extractor
        self.depth_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((8, 8)),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 8 * 8, 128),
            torch.nn.ReLU(),
        ).to(self.device)

        # GRU for temporal memory over depth observations
        self.gru = torch.nn.GRU(
            input_size=128, hidden_size=128,
            num_layers=1, batch_first=False,
        ).to(self.device)
        self.gru_hidden = torch.zeros(1, self.num_envs, 128, device=self.device)

        # Project GRU output to scene latent z_s (same dim as teacher's map_code)
        self.latent_projector = torch.nn.Linear(128, 32).to(self.device)

        # Student backbone: z_s(32) + proprio(62) + goal_body(3) -> c(5)
        student_input_dim = 32 + 62 + 3
        self.backbone_student = torch.nn.Sequential(
            torch.nn.Linear(student_input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 5),
        ).to(self.device)

    def compute_observations(self):
        """Student observation: depth -> CNN -> GRU -> z_s, proprio, goal."""
        low_env = self.low_level_env
        if low_env is None:
            return
        n = self.num_envs

        # Proprioceptive from low-level
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

        # Relative goal position in body frame
        robot_pos = low_env.root_states[:n, :3]
        robot_quat = low_env.base_quat[:n]
        delta_world = self.global_goal[:n] - robot_pos  # student uses final goal
        delta_body = quat_rotate_inverse(robot_quat, delta_world)

        # Depth image -> CNN -> GRU -> scene latent z_s
        depth_img = self._get_depth_images()  # (n, 1, 180, 320)
        depth_feat = self.depth_encoder(depth_img)  # (n, 128)
        depth_feat_seq = depth_feat.unsqueeze(0)    # (1, n, 128) for GRU
        gru_out, self.gru_hidden = self.gru(depth_feat_seq, self.gru_hidden)
        z_s = self.latent_projector(gru_out.squeeze(0))  # (n, 32)

        self.obs_buf = torch.cat([z_s, hl_prop, delta_body], dim=1)

        # Pad to match config num_observations
        if self.obs_buf.shape[1] < self.cfg.env.num_observations:
            pad = torch.zeros(n, self.cfg.env.num_observations - self.obs_buf.shape[1],
                              device=self.device)
            self.obs_buf = torch.cat([self.obs_buf, pad], dim=1)

    def _get_depth_images(self):
        """Acquire virtual depth images from simulation.

        Simplified: returns zero tensor. In production, this would use Isaac Gym's
        camera API to render depth from each env's viewpoint.
        """
        return torch.zeros(self.num_envs, 1, self.depth_h, self.depth_w,
                           device=self.device)

    def forward(self, obs_buf):
        """DAgger inference: obs -> command c + scene latent z_s."""
        c = self.backbone_student(obs_buf[:, :97])  # strip padding
        z_s = obs_buf[:, :32]  # first 32 dims are scene latent
        return c, z_s

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self.gru_hidden is not None:
            self.gru_hidden[:, env_ids] = 0.
