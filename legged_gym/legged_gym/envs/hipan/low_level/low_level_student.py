# legged_gym/envs/hipan/low_level/low_level_student.py
"""HiPAN low-level student policy piL_S: DAgger distillation from teacher.

Proprioceptive history (50 steps) -> Conv1D domain estimator -> zd_hat
                                     -> Conv1D motion estimator -> xm_hat
op_last(57) + c(5) + xm_hat(5) + zd_hat(32) -> backbone -> delta_q(12)

Distilled from LowLevelTeacher via DAgger supervised learning.
"""
import torch
from legged_gym.envs.hipan.low_level.low_level_teacher import LowLevelTeacher


class LowLevelStudent(LowLevelTeacher):
    """HiPAN low-level student piL_S: DAgger distillation, proprioceptive history -> joint actions"""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.proprio_history_len = 50

    def _init_buffers(self):
        super()._init_buffers()
        # 50-step proprioceptive history ring buffer
        self.proprio_history = torch.zeros(
            self.num_envs, self.proprio_history_len, 57,
            dtype=torch.float, device=self.device, requires_grad=False)
        self.history_ptr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Estimator e_d: 50-step history -> z_d_hat (domain latent)
        self.estimator_domain = torch.nn.Sequential(
            torch.nn.Conv1d(57, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
        ).to(self.device)

        # Estimator e_m: 10-step history -> x_m_hat (motion state)
        self.estimator_motion = torch.nn.Sequential(
            torch.nn.Conv1d(57, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 5),
        ).to(self.device)

        # Backbone b_S: op_last(57) + c(5) + x_m_hat(5) + z_d_hat(32) -> delta_q(12)
        self.backbone_student = torch.nn.Sequential(
            torch.nn.Linear(57 + 5 + 5 + 32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 12),
        ).to(self.device)

    def update_proprio_history(self):
        """Push current proprio_buf into ring buffer after each step."""
        op = self.proprio_buf  # (num_envs, 57), filled by compute_observations
        ptr = self.history_ptr
        self.proprio_history[torch.arange(self.num_envs, device=self.device), ptr] = op
        self.history_ptr = (ptr + 1) % self.proprio_history_len

    def get_history_window(self, window_len):
        """Get most recent window_len steps from ring buffer, time-aligned.

        Returns (num_envs, 57, window_len) for Conv1d input.
        """
        # ptr points to the next free slot (after the most recent write)
        # most recent entries are at (ptr-1) % N, (ptr-2) % N, ..., (ptr-window_len) % N
        idx = (self.history_ptr.unsqueeze(1) -
               torch.arange(window_len, 0, -1, device=self.device).unsqueeze(0))
        idx = idx % self.proprio_history_len
        batch_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, window_len)
        hist = self.proprio_history[batch_idx, idx]  # (num_envs, window_len, 57)
        return hist.transpose(1, 2)  # (num_envs, 57, window_len) for Conv1d

    def compute_observations(self):
        """Student observation: proprio history + commands, with estimator-inferred privilege."""
        # Fill current proprioceptive observation (same 57-dim as teacher)
        self.proprio_buf = torch.cat((
            self.dof_pos - self.default_dof_pos.squeeze(0),
            self.dof_vel * self.obs_scales.dof_vel,
            self.foot_pos.reshape(self.num_envs, -1),
            (self.contact_forces[:, self.feet_indices, 2] > 1.).float(),
            self.projected_gravity[:, :2],
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.last_actions,
        ), dim=-1)

        self.update_proprio_history()

        # Estimator inference from proprioceptive history
        hist_50 = self.get_history_window(self.proprio_history_len)  # (N, 57, 50)
        hist_10 = self.get_history_window(10)                         # (N, 57, 10)
        self.zd_hat = self.estimator_domain(hist_50)
        self.xm_hat = self.estimator_motion(hist_10)

        # Student backbone input: op_last + scaled commands + inferred privilege
        self.obs_buf = torch.cat((
            self.proprio_buf,
            self.commands * self.commands_scale,
            self.xm_hat,
            self.zd_hat,
        ), dim=-1)

    def forward(self, obs_buf):
        """DAgger inference: input obs_buf -> output delta_q + estimated domain latent.

        During DAgger training, obs_buf is from the stored dataset so zd_hat must be
        extracted from the observation itself (last 32 dims), not from self.zd_hat
        which reflects the current environment step.
        """
        delta_q = self.backbone_student(obs_buf)
        zd_hat = obs_buf[:, -32:]  # Last 32 dims are zd_hat
        return delta_q, zd_hat

    def reset_idx(self, env_ids):
        """Reset envs: clear proprioceptive history for reset envs."""
        super().reset_idx(env_ids)
        self.proprio_history[env_ids] = 0.
        self.history_ptr[env_ids] = 0

    def _prepare_reward_function(self):
        """Student uses DAgger MSE loss, not RL rewards."""
        pass
