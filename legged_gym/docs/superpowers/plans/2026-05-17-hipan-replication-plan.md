# HiPAN 仿真训练管线复刻 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 Isaac Gym + Go2 框架上构建 HiPAN 四阶段分层强化学习训练管线

**Architecture:** HiPAN 新模块位于 `envs/hipan/`，低层继承 `LeggedRobot`，高层继承 `BaseTask`。两层通过 5D 指令接口 `c=[vx,vy,ωz,h,θx]` 解耦。低层先收敛后嵌入高层做闭环训练。PGCL 课程 + DAgger 蒸馏照论文实现。

**Tech Stack:** Isaac Gym Preview 3, rsl_rl (PPO), PyTorch, NumPy

---

### Task 1: DAgger 蒸馏工具

**Files:**
- Create: `legged_gym/utils/dagger.py`

- [ ] **Step 1: 实现 DADataset 类 — 支持在线采集和教师标注**

```python
# legged_gym/utils/dagger.py
import torch
from torch.utils.data import Dataset
from typing import Callable

class DADataset(Dataset):
    """DAgger 数据集: 存储 (observation, teacher_action, teacher_latents) 三元组"""
    def __init__(self, max_size=100000):
        self.observations = []
        self.teacher_actions = []
        self.teacher_latents = []
        self.max_size = max_size

    def add(self, obs, action, latent=None):
        if len(self.observations) >= self.max_size:
            idx = torch.randint(0, len(self.observations), (1,)).item()
            self.observations[idx] = obs.cpu()
            self.teacher_actions[idx] = action.cpu()
            if latent is not None:
                self.teacher_latents[idx] = latent.cpu()
        else:
            self.observations.append(obs.cpu())
            self.teacher_actions.append(action.cpu())
            if latent is not None:
                self.teacher_latents.append(latent.cpu())
            else:
                self.teacher_latents.append(torch.zeros(1))

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.teacher_actions[idx], self.teacher_latents[idx]
```

- [ ] **Step 2: 实现 DAgger 训练循环**

```python
class DAggerTrainer:
    """DAgger 在线蒸馏训练器"""
    def __init__(self, teacher_policy, student_policy, optimizer,
                 device, latent_dim=32, batch_size=256, dagger_epochs=5):
        self.teacher = teacher_policy
        self.student = student_policy
        self.optimizer = optimizer
        self.device = device
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.dagger_epochs = dagger_epochs
        self.dataset = DADataset()

    def collect_and_label(self, env, student_policy_fn, teacher_inference_fn,
                          num_steps_per_env=24):
        """ rollout 学生策略 → 教师标注 → 加入数据集 """
        obs = env.get_observations()
        for _ in range(num_steps_per_env):
            with torch.no_grad():
                # 学生推理
                if isinstance(obs, tuple):
                    student_out, _ = student_policy_fn(obs)
                else:
                    student_out = student_policy_fn(obs)

                # 教师对同一观测进行标注
                teacher_out, teacher_latent = teacher_inference_fn(obs)

            # 存储 (观测, 教师动作, 教师隐变量)
            self.dataset.add(obs.detach().clone(),
                             teacher_out.detach().clone(),
                             teacher_latent.detach().clone() if teacher_latent is not None else None)

            # 学生动作推进环境
            if isinstance(student_out, tuple):
                student_out = student_out[0]
            obs, _, _, _, _ = env.step(student_out.detach())
            if isinstance(obs, tuple):
                obs = obs[0]

    def train_epoch(self):
        """一轮 DAgger 优化 """
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True)
        total_loss = 0.0
        for batch_obs, batch_act, batch_lat in dataloader:
            batch_obs = [b.to(self.device) for b in batch_obs] if isinstance(
                batch_obs, list) else batch_obs.to(self.device)
            batch_act = batch_act.to(self.device)
            batch_lat = batch_lat.to(self.device)

            pred_act, pred_lat = self.student(batch_obs)

            loss = torch.nn.functional.mse_loss(pred_act, batch_act)
            if batch_lat.abs().sum() > 0:
                loss = loss + torch.nn.functional.mse_loss(pred_lat, batch_lat)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, env, student_policy_fn, teacher_inference_fn,
              num_iterations=10, steps_per_iter=24):
        """完整 DAgger 训练: 交替 采集→训练 """
        for iteration in range(num_iterations):
            self.collect_and_label(env, student_policy_fn,
                                   teacher_inference_fn, steps_per_iter)
            for _ in range(self.dagger_epochs):
                epoch_loss = self.train_epoch()
        return epoch_loss
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/utils/dagger.py
git commit -m "feat: 添加 DAgger 在线蒸馏工具 (DADataset + DAggerTrainer)"
```

---

### Task 2: LeggedRobot 基类钩子

**Files:**
- Modify: `legged_gym/envs/base/legged_robot.py`

HiPAN 低层需要重写 `compute_observations`、`compute_reward`、`_init_buffers` 等，这些方法在 `LeggedRobot` 中已有。为使子类能干净地复用底层（PD控制、URDF加载、域随机化回调）而不受基类观察/奖励逻辑约束，加入少量钩子。

- [ ] **Step 1: 将 `_get_env_origins` 中对 `mesh_type` 的判断抽取为可重写方法**

在 `legged_robot.py` 的 `_get_env_origins` 方法中 (L722-747)，当前的 if/elif 链只处理 `heightfield`/`trimesh` 和 `else`。添加 `elif` 分支留给子类扩展：

```python
# 在 _get_env_origins 方法中, 于 L737 的 else 之前插入:
elif self.cfg.terrain.mesh_type in ["competition"]:
    self.custom_origins = False
    self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
    self._set_custom_env_origins()  # 子类可重写
```

在 `LeggedRobot` 类中增加空钩子：

```python
def _set_custom_env_origins(self):
    """子类重写以设置自定义出生点逻辑。默认无操作。"""
    pass
```

- [ ] **Step 2: 将 `create_sim` 中的 terrain mesh_type 分支改为可扩展**

在 `create_sim` 方法 (L243-258) 的 elif 链末尾加：

```python
elif mesh_type == 'wfc':
    self._create_wfc_terrain()
```

并在 `LeggedRobot` 中增加空实现：

```python
def _create_wfc_terrain(self):
    raise NotImplementedError("WFC terrain not implemented in base class")
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/envs/base/legged_robot.py
git commit -m "feat: LeggedRobot 基类添加 _set_custom_env_origins 和 _create_wfc_terrain 钩子"
```

---

### Task 3: HiPAN 任务注册骨架

**Files:**
- Create: `legged_gym/envs/hipan/__init__.py`
- Modify: `legged_gym/envs/__init__.py`

- [ ] **Step 1: 创建 hipan 包 `__init__.py`**

```python
# legged_gym/envs/hipan/__init__.py
# HiPAN: Hierarchical Posture-Adaptive Navigation 仿真训练管线
from .low_level.low_level_teacher import LowLevelTeacher
from .low_level.low_level_student import LowLevelStudent
from .low_level.low_level_config import LowLevelCfg, LowLevelCfgPPO
from .high_level.high_level_teacher import HighLevelTeacher
from .high_level.high_level_student import HighLevelStudent
from .high_level.high_level_config import HighLevelCfg, HighLevelCfgPPO
```

- [ ] **Step 2: 在 `envs/__init__.py` 注册 Hipan 任务**

```python
# 在 legged_gym/envs/__init__.py 末尾添加:
from legged_gym.envs.hipan.low_level.low_level_teacher import LowLevelTeacher
from legged_gym.envs.hipan.low_level.low_level_student import LowLevelStudent
from legged_gym.envs.hipan.low_level.low_level_config import LowLevelCfg, LowLevelCfgPPO
from legged_gym.envs.hipan.high_level.high_level_teacher import HighLevelTeacher
from legged_gym.envs.hipan.high_level.high_level_student import HighLevelStudent
from legged_gym.envs.hipan.high_level.high_level_config import HighLevelCfg, HighLevelCfgPPO

# 低层教师训练任务
task_registry.register("hipan_low_teacher", LowLevelTeacher, LowLevelCfg(), LowLevelCfgPPO())
# 低层学生 (通过专用脚本 train_low_student.py 启动 DAgger, 不在标准 PPO 流程中)
task_registry.register("hipan_low_student", LowLevelStudent, LowLevelCfg(), LowLevelCfgPPO())
# 高层教师训练任务
task_registry.register("hipan_high_teacher", HighLevelTeacher, HighLevelCfg(), HighLevelCfgPPO())
# 高层学生 (同低层学生, 通过 DAgger 脚本启动)
task_registry.register("hipan_high_student", HighLevelStudent, HighLevelCfg(), HighLevelCfgPPO())
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/envs/hipan/__init__.py legged_gym/envs/__init__.py
git commit -m "feat: 注册 HiPAN 四任务到 task_registry"
```

---

### Task 4: 低层配置 LowLevelCfg

**Files:**
- Create: `legged_gym/envs/hipan/low_level/low_level_config.py`

- [ ] **Step 1: 创建低层环境配置**

完全照 HiPAN Table III 奖励 + 5D 指令空间 + Grid-Adaptive Curriculum 参数。继承 `LeggedRobotCfg` / `LeggedRobotCfgPPO`。

```python
# legged_gym/envs/hipan/low_level/low_level_config.py
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class LowLevelCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 62   # op(57) + c(5) (teacher), 学生会有单独配置
        num_privileged_obs = None
        num_actions = 12
        episode_length_s = 20
        send_timeouts = True

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'      # stairs/holes/slopes/flat 混合地形
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
        heading_command = False  # HiPAN 不使用 heading 模式, 直接用 ωz

        class ranges:
            lin_vel_x = [0.0, 0.5]   # Grid-Adaptive Curriculum 初始范围, 逐步扩展
            lin_vel_y = [0.0, 0.3]
            ang_vel_yaw = [0.0, 0.5]
            height = [0.15, 0.35]
            roll = [0.0, 0.3]

        # Grid-Adaptive Curriculum 参数
        grid_adaptive = True
        ga_threshold = 0.8          # 跟踪达标比例阈值
        ga_expand_step = 0.1        # 每次扩展步长
        ga_max_ranges = {
            'lin_vel_x': [-1.5, 1.5],
            'lin_vel_y': [-1.0, 1.0],
            'ang_vel_yaw': [-1.5, 1.5],
            'height': [0.1, 0.4],
            'roll': [-1.0, 1.0],
        }

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
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
        stiffness = {'joint': 20}
        damping = {'joint': 0.5}
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
        only_positive_rewards = False  # HiPAN 使用负奖励
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        tracking_sigma_vel = 0.25
        tracking_sigma_yaw = 0.25
        tracking_sigma_height = 0.025  # 0.0025*10 (dt补偿前的值)
        tracking_sigma_roll = 0.05

        class scales:
            velocity_tracking = 0.4
            yaw_tracking = 0.2
            height_tracking = 0.2
            roll_tracking = 0.2
            action_rate = -0.01
            smooth_action = -0.1
            body_orientation = -0.5
            body_velocity = -0.2
            smooth_joint_vel = -0.001
            smooth_joint_acc = -0.0001
            torque_usage = -0.0001
            joint_limit = -10.0
            collision = -10.0

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
```

- [ ] **Step 2: Commit**

```bash
git add legged_gym/envs/hipan/low_level/low_level_config.py
git commit -m "feat: 创建低层配置 LowLevelCfg (5D指令 + HiPAN Table III 奖励 + Grid-Adaptive Curriculum)"
```

---

### Task 5: 低层教师 LowLevelTeacher

**Files:**
- Create: `legged_gym/envs/hipan/low_level/low_level_teacher.py`

- [ ] **Step 1: 创建 LowLevelTeacher 类骨架 — 继承 LeggedRobot, 重写核心方法**

```python
# legged_gym/envs/hipan/low_level/low_level_teacher.py
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.helpers import class_to_dict

class LowLevelTeacher(LeggedRobot):
    """HiPAN 低层教师策略 πL_T: PPO训练, 特权信息 → 关节动作 Δq"""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        # HiPAN 低层观察维度: op(57) + c(5)
        self.num_low_obs = 57
        self.num_command = 5

    def _init_buffers(self):
        super()._init_buffers()
        # 覆盖基类 commands 为 5D
        self.commands = torch.zeros(self.num_envs, self.num_command,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel,
                                            self.obs_scales.lin_vel,
                                            self.obs_scales.ang_vel,
                                            1.0, 1.0],
                                           device=self.device, requires_grad=False)

        # 特权域参数编码器
        self.domain_encoder = torch.nn.Sequential(
            torch.nn.Linear(187 + 5, 128),  # height_samples(187) + params(5)
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
        ).to(self.device)

        # 主干网络: op + c + xm + zd → Δq
        input_dim = 57 + 5 + 5 + 32  # op + c + xm + zd
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 12),  # Δq
        ).to(self.device)

    def _resample_commands(self, env_ids):
        """采样 5D 指令 [vx, vy, ωz, h, θx]"""
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(
            self.command_ranges["height"][0],
            self.command_ranges["height"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 4] = torch_rand_float(
            self.command_ranges["roll"][0],
            self.command_ranges["roll"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)

    def _get_domain_params(self):
        """构建特权域参数向量: 高度图 + 摩擦/质量/电机 """
        if self.cfg.terrain.measure_heights:
            height_feat = self.measured_heights
        else:
            height_feat = torch.zeros(self.num_envs, 187, device=self.device)
        friction = self.friction_coeffs.squeeze(-1) if hasattr(self, 'friction_coeffs') else torch.ones(self.num_envs, device=self.device)
        mass = torch.ones(self.num_envs, device=self.device)
        motor = getattr(self, 'motor_strength', torch.ones(2, self.num_envs, self.num_dof, device=self.device))
        motor_mean = motor.mean(dim=[0, 2])
        params = torch.stack([friction, mass, motor_mean[0], motor_mean[1],
                              torch.zeros(self.num_envs, device=self.device)], dim=1)
        return torch.cat([height_feat, params], dim=1)

    def compute_observations(self):
        """观察: op(57) + c(5), 教师内部使用特权 xm + zd"""
        # op: proprioceptive
        self.proprio_buf = torch.cat((
            self.dof_pos - self.default_dof_pos.squeeze(0),  # 12
            self.dof_vel * self.obs_scales.dof_vel,           # 12
            self.foot_pos.reshape(self.num_envs, -1),          # 12
            self.contact_forces[:, self.feet_indices, 2] > 1., # 4
            self.projected_gravity[:, :2],                     # 2
            self.base_ang_vel * self.obs_scales.ang_vel,      # 3
            self.last_actions,                                  # 12
        ), dim=-1)  # 57维

        # 特权运动状态 xm
        h_body = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1, keepdim=True)
        self.xm = torch.cat((
            self.base_lin_vel,   # v_B (3)
            h_body,               # h_B  (1)
            torch.atan2(self.projected_gravity[:, 0],
                        self.projected_gravity[:, 2]).unsqueeze(1),  # θ^W_x (1)
        ), dim=-1)  # 5维

        # 域参数隐向量
        domain_input = self._get_domain_params()
        self.zd = self.domain_encoder(domain_input)

        # 特权的完整输入 (教师内部使用, RL 算法从 self.obs_buf 读)
        self.obs_buf = torch.cat((
            self.proprio_buf,
            self.commands * self.commands_scale,
            self.xm,
            self.zd,
        ), dim=-1)

    def _compute_torques(self, actions):
        """PD控制: 基础类是位置型, 需要将 actions 视为关节偏移量 Δq """
        self.actions = actions
        target_pos = self.default_dof_pos.squeeze(0) + actions * self.cfg.control.action_scale
        torques = self.p_gains * (target_pos - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _prepare_reward_function(self):
        """HiPAN 低层 12 项奖励"""
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            name_lower = '_reward_' + name
            if hasattr(self, name_lower) and name != "termination":
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self, name_lower))
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float,
                              device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    # ---------- 奖励函数 (HiPAN Table III) ----------
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
            self.actions - 2 * self.last_actions + self.last_dof_vel * 0), dim=1)

    def _reward_body_orientation(self):
        return torch.abs(self.projected_gravity[:, 1])

    def _reward_body_velocity(self):
        return torch.square(self.base_lin_vel[:, 2]) + torch.sum(
            torch.square(self.base_ang_vel[:, :2]), dim=1)

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
```

- [ ] **Step 2: Commit**

```bash
git add legged_gym/envs/hipan/low_level/low_level_teacher.py
git commit -m "feat: 低层教师 LowLevelTeacher (特权观察 + 5D指令PD控制器 + 12项HiPAN奖励)"
```

---

### Task 6: 低层学生 LowLevelStudent

**Files:**
- Create: `legged_gym/envs/hipan/low_level/low_level_student.py`
- Create: `legged_gym/scripts/train_low_student.py`

- [ ] **Step 1: 创建 LowLevelStudent — 带估计器 and 50步历史环缓冲区**

```python
# legged_gym/envs/hipan/low_level/low_level_student.py
import torch
import numpy as np
from legged_gym.envs.hipan.low_level.low_level_teacher import LowLevelTeacher

class LowLevelStudent(LowLevelTeacher):
    """HiPAN 低层学生策略 πL_S: DAgger蒸馏, 本体感知历史→关节动作 Δq"""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.proprio_history_len = 50

    def _init_buffers(self):
        # 调用 LowLevelTeacher 的 _init_buffers (不是 LeggedRobot)
        super()._init_buffers()
        # 50步本体感知历史环缓冲区
        self.proprio_history = torch.zeros(
            self.num_envs, self.proprio_history_len, 57,
            dtype=torch.float, device=self.device, requires_grad=False)
        self.history_ptr = torch.zeros(self.num_envs, dtype=torch.long,
                                        device=self.device)

        # 估计器 e_d: 50步历史 → ẑ_d (域参数隐向量)
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

        # 估计器 e_m: 10步历史 → x̂_m (运动状态)
        self.estimator_motion = torch.nn.Sequential(
            torch.nn.Conv1d(57, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 5),
        ).to(self.device)

        # 主干网络 b_S: op_last + c + x̂_m + ẑ_d → Δq
        self.backbone_student = torch.nn.Sequential(
            torch.nn.Linear(57 + 5 + 5 + 32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 12),
        ).to(self.device)

    def update_proprio_history(self):
        """每个 step 后将当前 op 压入历史环缓冲区"""
        op = self.proprio_buf  # (num_envs, 57), 在 compute_observations 中填充
        ptr = self.history_ptr
        self.proprio_history[torch.arange(self.num_envs, device=self.device), ptr] = op
        self.history_ptr = (ptr + 1) % self.proprio_history_len

    def get_history_window(self, window_len):
        """从环缓冲区取最近 window_len 步的历史, 对齐时间"""
        idx = (self.history_ptr.unsqueeze(1) -
               torch.arange(window_len, 0, -1, device=self.device).unsqueeze(0))
        idx = idx % self.proprio_history_len
        batch_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, window_len)
        hist = self.proprio_history[batch_idx, idx]  # (num_envs, window_len, 57)
        # 转置为 (num_envs, 57, window_len) 供 Conv1d 使用
        return hist.transpose(1, 2)

    def compute_observations(self):
        """学生观察: 本体历史 + 指令, 配估计器推断特权"""
        # 填充当前 op
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

        # 估计器推理
        hist_50 = self.get_history_window(self.proprio_history_len)  # (N, 57, 50)
        hist_10 = self.get_history_window(10)                         # (N, 57, 10)
        self.zd_hat = self.estimator_domain(hist_50)
        self.xm_hat = self.estimator_motion(hist_10)

        # 学生的主干输入
        op_last = self.proprio_buf
        self.obs_buf = torch.cat((
            op_last,
            self.commands * self.commands_scale,
            self.xm_hat,
            self.zd_hat,
        ), dim=-1)

    def forward(self, obs_buf):
        """供 DAgger 调用: 输入 obs_buf → 输出 Δq + latents"""
        # obs_buf 已在 compute_observations 中组装好
        # RL 算法通常通过 step() 使用, 这里提供独立 forward 供 DAgger
        delta_q = self.backbone_student(obs_buf)
        latent = self.zd_hat  # 当前帧的隐变量作为 latent
        return delta_q, latent

    def reset_idx(self, env_ids):
        """重置时清零历史"""
        super().reset_idx(env_ids)
        self.proprio_history[env_ids] = 0.
        self.history_ptr[env_ids] = 0

    def _prepare_reward_function(self):
        # 学生蒸馏通过 DAgger, 不需要 RL 奖励; 但保留兼容性
        pass
```

- [ ] **Step 2: 创建低层学生 DAgger 训练脚本**

```python
# legged_gym/scripts/train_low_student.py
import os
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.hipan.low_level.low_level_teacher import LowLevelTeacher
from legged_gym.envs.hipan.low_level.low_level_student import LowLevelStudent
from legged_gym.utils.dagger import DAggerTrainer

def train_low_student(args):
    # 1. 加载已训练好的低层教师
    teacher_cfg, train_cfg = task_registry.get_cfgs(name="hipan_low_teacher")
    teacher_cfg.env.num_envs = 300
    teacher_env, _ = task_registry.make_env(name="hipan_low_teacher", args=args, env_cfg=teacher_cfg)

    # 加载教师 checkpoint
    train_cfg.runner.resume = True
    teacher_runner, train_cfg = task_registry.make_alg_runner(
        env=teacher_env, name="hipan_low_teacher", args=args, train_cfg=train_cfg)
    teacher_policy = teacher_runner.get_inference_policy(device=teacher_env.device)

    # 2. 创建学生环境
    student_cfg, _ = task_registry.get_cfgs(name="hipan_low_student")
    student_cfg.env.num_envs = 300
    student_env, _ = task_registry.make_env(name="hipan_low_student", args=args, env_cfg=student_cfg)

    # 3. 构建 DAgger
    student_model = student_env.backbone_student  # 学生主干
    optimizer = torch.optim.Adam(
        list(student_env.estimator_domain.parameters()) +
        list(student_env.estimator_motion.parameters()) +
        list(student_env.backbone_student.parameters()),
        lr=1e-4
    )

    dagger = DAggerTrainer(
        teacher_policy=teacher_policy,
        student_policy=lambda obs: (student_env.forward(obs), None),
        optimizer=optimizer,
        device=student_env.device,
    )

    # 4. 定义标注函数
    def teacher_label_fn(obs_buf):
        # 低层教师输入: obs_buf 已有 op + c + xm + zd
        action = teacher_policy(obs_buf)
        latent = student_env.zd  # 教师计算的 zd
        return action, latent

    def student_policy_fn(obs_buf):
        return student_env.forward(obs_buf)

    # 5. DAgger 训练循环
    for iteration in range(20):
        dagger.collect_and_label(student_env, student_policy_fn, teacher_label_fn, 48)
        loss = dagger.train_epoch()
        print(f"DAgger iter {iteration}: loss={loss:.6f}")

    # 6. 保存
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'logs', 'hipan_low_student', 'student_checkpoint.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'student_backbone': student_env.backbone_student.state_dict(),
        'estimator_domain': student_env.estimator_domain.state_dict(),
        'estimator_motion': student_env.estimator_motion.state_dict(),
    }, save_path)
    print(f"Student saved to {save_path}")

if __name__ == '__main__':
    args = get_args()
    args.task = "hipan_low_student"
    train_low_student(args)
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/envs/hipan/low_level/low_level_student.py legged_gym/scripts/train_low_student.py
git commit -m "feat: 低层学生 LowLevelStudent (1D Conv估计器 + 50步历史 + DAgger训练脚本)"
```

---

### Task 7: WFC 地形生成

**Files:**
- Create: `legged_gym/envs/hipan/terrain/__init__.py`
- Create: `legged_gym/envs/hipan/terrain/wfc_terrain.py`

- [ ] **Step 1: 实现 WFC 地形生成**

WFC 算法使用预定义 tile 之间的邻接约束组装地形。简化实现：定义若干 tile 类型 → 随机排列 → 拼接。

```python
# legged_gym/envs/hipan/terrain/wfc_terrain.py
import numpy as np
from isaacgym.terrain_utils import (
    SubTerrain, convert_heightfield_to_trimesh,
    sloped_terrain, pyramid_sloped_terrain, random_uniform_terrain,
    discrete_obstacles_terrain, wave_terrain, stairs_terrain,
    flat_terrain
)

class WFCTerrain:
    """WFC 风格地形生成: 随机排列 tile 类型生成非结构化 3D 环境 """

    TILE_TYPES = ['flat', 'rough_slope', 'rough', 'obstacles', 'wave',
                  'stairs_up', 'stairs_down', 'pillars', 'mixed']

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_rows = cfg.num_rows
        self.num_cols = cfg.num_cols
        self.tile_width = int(cfg.terrain_width / cfg.horizontal_scale)
        self.tile_length = int(cfg.terrain_length / cfg.horizontal_scale)
        self.horizontal_scale = cfg.horizontal_scale
        self.vertical_scale = cfg.vertical_scale

        # 总尺寸: 地图 + 边界
        self.border = int(cfg.border_size / cfg.horizontal_scale)
        self.tot_rows = int(self.num_rows * self.tile_length) #+ 2 * self.border
        self.tot_cols = int(self.num_cols * self.tile_width) #+ 2 * self.border

        self.heightsamples = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self._generate()
        self.vertices, self.triangles = convert_heightfield_to_trimesh(
            self.heightsamples,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            slope_threshold=1.5
        )

    def _new_sub_terrain(self):
        return SubTerrain(width=self.tile_width, length=self.tile_length,
                          vertical_scale=self.vertical_scale,
                          horizontal_scale=self.horizontal_scale)

    def _generate_tile(self, tile_type, terrain):
        """根据 tile 类型生成高度场"""
        if tile_type == 'flat':
            return sloped_terrain(terrain, slope=0.0).height_field_raw
        elif tile_type == 'rough_slope':
            slope = np.random.uniform(-0.5, 0.5)
            return pyramid_sloped_terrain(terrain, slope=slope).height_field_raw
        elif tile_type == 'rough':
            return random_uniform_terrain(terrain, min_height=-0.1,
                                          max_height=0.1, step=0.15,
                                          downsampled_scale=0.5).height_field_raw
        elif tile_type == 'obstacles':
            return discrete_obstacles_terrain(terrain, max_height=0.2,
                                              min_size=0.5, max_size=3.,
                                              num_rects=15).height_field_raw
        elif tile_type == 'wave':
            return wave_terrain(terrain, num_waves=2.,
                                amplitude=np.random.uniform(0.5, 1.5)).height_field_raw
        elif tile_type == 'stairs_up':
            return stairs_terrain(terrain, step_width=0.5,
                                  step_height=np.random.uniform(0.05, 0.2)).height_field_raw
        elif tile_type == 'stairs_down':
            return stairs_terrain(terrain, step_width=0.5, step_height=-0.15).height_field_raw
        elif tile_type == 'pillars':
            return discrete_obstacles_terrain(terrain, max_height=0.4,
                                              min_size=0.3, max_size=0.5,
                                              num_rects=30).height_field_raw
        elif tile_type == 'mixed':
            return random_uniform_terrain(terrain, min_height=-0.2,
                                          max_height=0.3, step=0.2,
                                          downsampled_scale=0.4).height_field_raw
        else:
            return sloped_terrain(terrain, slope=0.0).height_field_raw

    def _generate(self):
        """随机排列 tile 生成完整地图 """
        n_tiles = self.num_rows * self.num_cols
        tiles = np.random.choice(self.TILE_TYPES, size=n_tiles, replace=True)
        for idx, tile_type in enumerate(tiles):
            row = idx // self.num_cols
            col = idx % self.num_cols
            terrain = self._new_sub_terrain()
            hf = self._generate_tile(tile_type, terrain)
            r_start = int(row * self.tile_length)
            c_start = int(col * self.tile_width)
            self.heightsamples[r_start:r_start+self.tile_length,
                               c_start:c_start+self.tile_width] = hf

    def get_height_samples(self):
        return torch.tensor(self.heightsamples).view(self.tot_rows, self.tot_cols)
```

- [ ] **Step 2: Terrain `__init__.py`**

```python
# legged_gym/envs/hipan/terrain/__init__.py
from .wfc_terrain import WFCTerrain
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/envs/hipan/terrain/
git commit -m "feat: WFC 地形生成 (9种tile类型随机排列组装非结构化环境)"
```

---

### Task 8: PGCL 课程管理器

**Files:**
- Create: `legged_gym/envs/hipan/pgcl.py`

- [ ] **Step 1: 实现 PGCL 课程**

```python
# legged_gym/envs/hipan/pgcl.py
import numpy as np
import torch

class PGCLManager:
    """Path-Guided Curriculum Learning: 沿全局路径渐进减少子目标 """

    def __init__(self, initial_d=1.0, d_step=1.0, path_planner=None):
        self.initial_d = initial_d
        self.d_step = d_step
        self.current_d = initial_d
        self.level = 0
        self.path_planner = path_planner  # A* 规划器 (后续 Task 中实现)

    def generate_subgoals(self, start, goal, privileged_map):
        """沿全局路径每 d 米放置一个子目标 """
        if self.path_planner is None:
            return [goal]

        path = self.path_planner.plan(start, goal, privileged_map)
        if len(path) < 2:
            return [goal]

        subgoals = []
        accumulated_dist = 0.0
        last_point = path[0]
        subgoals.append(last_point)

        for point in path[1:]:
            seg_dist = np.linalg.norm(np.array(point) - np.array(last_point))
            accumulated_dist += seg_dist
            last_point = point
            if accumulated_dist >= self.current_d:
                subgoals.append(point)
                accumulated_dist = 0.0

        if subgoals[-1] != goal:
            subgoals.append(goal)
        return subgoals

    def advance_level(self):
        """全部子目标到达 → 升级"""
        self.current_d += self.d_step
        self.level += 1
        return self.current_d

    def is_final_level(self, path_length):
        """最终 level: d > 路径总长 → 只剩起始→终点 """
        return self.current_d > path_length

    def get_state_dict(self):
        return {'level': self.level, 'current_d': self.current_d}
```

- [ ] **Step 2: A* 路径规划器 (带 Generalized Voronoi Diagram 简化实现)**

```python
# legged_gym/envs/hipan/pgcl.py (追加)
import heapq

class AStarPlanner:
    """A* 路径规划器, 在 2D 占据栅格地图上搜索 """
    def __init__(self, resolution=0.1, obstacle_threshold=0.1):
        self.resolution = resolution
        self.obstacle_threshold = obstacle_threshold

    def plan(self, start, goal, height_map):
        """在高度图上规划路径, 返回 waypoint 列表"""
        h, w = height_map.shape
        start_rc = self._world_to_grid(start, h, w)
        goal_rc = self._world_to_grid(goal, h, w)

        # 障碍物栅格: 高度变化 > threshold 的格子
        grad_y, grad_x = np.gradient(height_map)
        obstacle_mask = (np.abs(grad_y) + np.abs(grad_x)) > self.obstacle_threshold

        open_set = [(0, start_rc)]
        came_from = {}
        g_score = {start_rc: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_rc:
                # 重建路径
                path = [goal]
                while current in came_from:
                    current = came_from[current]
                    path.append(self._grid_to_world(current, h, w))
                path.reverse()
                return path

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                neighbor = (current[0]+dr, current[1]+dc)
                if not (0 <= neighbor[0] < h and 0 <= neighbor[1] < w):
                    continue
                if obstacle_mask[neighbor]:
                    continue
                tentative_g = g_score[current] + (1.414 if dr!=0 and dc!=0 else 1.0)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal_rc)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current
        return [start, goal]  # 找不到路径, 直接连直线

    def _world_to_grid(self, pos, h, w):
        r = int(pos[1] / self.resolution)
        c = int(pos[0] / self.resolution)
        return (np.clip(r, 0, h-1), np.clip(c, 0, w-1))

    def _grid_to_world(self, grid, h, w):
        return [grid[1] * self.resolution, grid[0] * self.resolution, 0.0]

    def _heuristic(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
```

- [ ] **Step 3: Intrinsic Reward 状态计数器**

```python
# legged_gym/envs/hipan/pgcl.py (追加)
class StateCountExploration:
    """基于离散网格的状态访问计数, 提供 Intrinsic Reward """
    def __init__(self, resolution=0.1, grid_size=(200, 200)):
        self.resolution = resolution
        self.counts = np.zeros(grid_size)

    def get_count(self, pos_xy):
        """获取位置 (x, y) 的访问计数 """
        idx = self._pos_to_idx(pos_xy)
        return self.counts[idx]

    def increment(self, pos_xy):
        idx = self._pos_to_idx(pos_xy)
        self.counts[idx] += 1

    def get_intrinsic_reward(self, pos_xy):
        """r = 1 / sqrt(count)"""
        count = self.get_count(pos_xy)
        if count < 1:
            return 1.0
        return 1.0 / np.sqrt(count)

    def _pos_to_idx(self, pos_xy):
        x_idx = int(pos_xy[0] / self.resolution)
        y_idx = int(pos_xy[1] / self.resolution)
        x_idx = np.clip(x_idx, 0, self.counts.shape[1] - 1)
        y_idx = np.clip(y_idx, 0, self.counts.shape[0] - 1)
        return (y_idx, x_idx)
```

- [ ] **Step 4: Commit**

```bash
git add legged_gym/envs/hipan/pgcl.py
git commit -m "feat: PGCL 课程管理器 + A* 规划器 + Intrinsic Reward 状态计数"
```

---

### Task 9: 高层配置 HighLevelCfg

**Files:**
- Create: `legged_gym/envs/hipan/high_level/high_level_config.py`

- [ ] **Step 1: 创建高层环境配置**

```python
# legged_gym/envs/hipan/high_level/high_level_config.py
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HighLevelCfg(LeggedRobotCfg):
    """HiPAN 高层导航策略配置: 10Hz, WFC地形, PGCL课程, 双地图感知"""

    class env(LeggedRobotCfg.env):
        num_envs = 1024
        num_observations = 256  # 地图编码后 + proprio + goal
        num_actions = 5   # [vx, vy, ωz, h, θx]
        episode_length_s = 120  # 长 horizon 导航
        send_timeouts = True

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'wfc'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 2.0
        curriculum = False
        measure_heights = True
        terrain_length = 20.
        terrain_width = 20.
        num_rows = 5
        num_cols = 5
        terrain_proportions = [0.0] * 5

    class nav:
        """导航相关参数 """
        high_level_freq = 10   # Hz
        low_level_steps_per_high = 5  # 100ms = 5 * 0.02s
        desired_speed_range = [0.3, 1.2]  # m/s
        goal_arrival_threshold = 0.1       # m
        subgoal_arrival_threshold = 0.1    # m

        # 地图参数
        map_3d_resolution = 0.1
        map_3d_range = [-0.5, 0.5]      # 局部 ±0.5m
        map_3d_size = [14, 11, 11]      # [height, width, depth]
        map_2d5_resolution = 0.1
        map_2d5_range = [-1.0, 1.0]     # 局部 ±1.0m
        map_2d5_size = [31, 21]

        # PGCL
        pgcl_initial_d = 1.0
        pgcl_d_step = 1.0

    class commands(LeggedRobotCfg.commands):
        num_commands = 5
        heading_command = False
        class ranges:
            lin_vel_x = [-1.5, 1.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.5, 1.5]
            height = [0.1, 0.4]
            roll = [-1.0, 1.0]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
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
        stiffness = {'joint': 20}
        damping = {'joint': 0.5}
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
        randomize_friction = False
        friction_range = [0.7, 1.2]
        randomize_base_mass = False
        added_mass_range = [0.0, 3.0]
        randomize_base_com = False
        added_com_range = [-0.1, 0.1]
        randomize_motor = False
        motor_strength_range = [0.9, 1.1]
        push_robots = False

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.3

        class scales:
            goal_arrival = 5.0
            state_count = 0.5
            desired_speed = 0.25
            command_rate = -0.1
            smooth_command = -0.1
            tracking_error = -0.2
            body_velocity = -0.1
            nominal_posture = -0.04
            command_limit = -2.5
            collision = -2.5

    class normalization:
        class obs_scales:
            dof_pos = 1.0
            dof_vel = 0.05
            ang_vel = 0.25
            lin_vel = 2.0
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise(LeggedRobotCfg.noise):
        add_noise = False

    class viewer:
        pos = [25, 0, 15]
        lookat = [22., 10, 2.]

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        gravity = [0., 0., -9.81]
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1


class HighLevelCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'hipan_high_teacher'
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 100   # 高层 10Hz, 100步 = 10s rollout
        max_iterations = 8000

        save_interval = 50
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
```

- [ ] **Step 2: Commit**

```bash
git add legged_gym/envs/hipan/high_level/high_level_config.py
git commit -m "feat: 高层配置 HighLevelCfg (10Hz, WFC, PGCL, 双地图感知)"
```

---

### Task 10: 高层教师 HighLevelTeacher

**Files:**
- Create: `legged_gym/envs/hipan/high_level/high_level_teacher.py`

- [ ] **Step 1: 创建高层教师 — 内嵌低层学生**

```python
# legged_gym/envs/hipan/high_level/high_level_teacher.py
import torch
import numpy as np
from legged_gym.envs.base.base_task import BaseTask
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.hipan.terrain.wfc_terrain import WFCTerrain
from legged_gym.envs.hipan.pgcl import PGCLManager, AStarPlanner, StateCountExploration
from legged_gym.utils.helpers import class_to_dict


class HighLevelTeacher(BaseTask):
    """HiPAN 高层教师策略 πH_T: 双地图特权感知 → 5D导航指令, 内嵌冻结低层学生 """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.low_level_sim = None     # 低层仿真句柄 (通过 make_alg_runner 注入)
        self.low_level_policy = None  # 冻结的低层学生策略
        self.cfg = cfg
        self.sim_params = sim_params
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def set_low_level(self, low_level_env, low_level_policy):
        """注入已训练的低层学生 """
        self.low_level_env = low_level_env
        self.low_level_policy = low_level_policy

    def _init_buffers(self):
        # 命令缓冲区
        self.commands = torch.zeros(self.num_envs, 5, dtype=torch.float,
                                     device=self.device, requires_grad=False)
        self.last_command = torch.zeros_like(self.commands)
        self.second_last_command = torch.zeros_like(self.commands)

        # PGCL 管理 (每 env 独立)
        self.pgcl_managers = []
        self.state_explorer = StateCountExploration(resolution=0.1)
        self.path_planner = AStarPlanner()

        # 双地图编码器
        self.map3d_encoder = torch.nn.Sequential(
            torch.nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 16),
        ).to(self.device)

        self.map2d_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 16),
        ).to(self.device)

        # 主干网络
        backbone_input_dim = 16 + 16 + 62 + 3  # map_enc + op(62) + subgoal(3)
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(backbone_input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 5),  # 输出 c = [vx, vy, ωz, h, θx]
        ).to(self.device)

        # 目标/子目标相关
        self.global_goal = torch.zeros(self.num_envs, 3, device=self.device)
        self.current_subgoal = torch.zeros(self.num_envs, 3, device=self.device)
        self.pgcl_levels = torch.zeros(self.num_envs, dtype=torch.int32, device='cpu')

        # 仿真时间为高层 10Hz
        self.high_dt = 0.1

    def create_sim(self):
        # 高层不创建机器人仿真; WFC 地形在 generate_maps 中处理
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                        self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self.gym.prepare_sim(self.sim)

    def _create_ground_plane(self):
        from isaacgym import gymapi
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 0.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def build_privileged_maps(self, wfc_terrain, robot_positions):
        """为每个 env 构建局部 M_3D 和 M_2.5D"""
        height_samples = wfc_terrain.get_height_samples()  # (H, W)
        map_3d_size = self.cfg.nav.map_3d_size
        map_2d5_size = self.cfg.nav.map_2d5_size
        resolution = self.cfg.nav.map_3d_resolution

        batch_M3D = torch.zeros(self.num_envs, 1, *map_3d_size, device=self.device)
        batch_M2D = torch.zeros(self.num_envs, 1, *map_2d5_size, device=self.device)

        for i in range(self.num_envs):
            rx, ry = robot_positions[i, 0].item(), robot_positions[i, 1].item()
            # 裁剪局部地图
            # (简化实现: 完整地图需要根据机器人位置裁剪;
            #  实际实现需要更多坐标变换, 此处展示接口)
            batch_M2D[i, 0, :, :] = 0.5  # placeholder
            batch_M3D[i, 0, :, :, :] = 0.0  # placeholder
        return batch_M3D, batch_M2D

    def step(self, actions):
        """高层 step: 输出 5D c → 低层执行 5 步 → 返回高层状态 """
        c = torch.clip(actions, -1.0, 1.0)  # 视为归一化指令
        c_scaled = c.clone()
        c_scaled[:, 0] *= 1.5  # vx 范围
        c_scaled[:, 1] *= 1.0  # vy
        c_scaled[:, 2] *= 1.5  # ωz
        c_scaled[:, 3] = (c[:, 3] + 1) / 2 * 0.3 + 0.1  # h: [0.1, 0.4]
        c_scaled[:, 4] *= 1.0  # θx

        # 嵌入低层: 在 100ms 内固定指令, 跑 5 步低层
        low_env = self.low_level_env
        low_env.commands[:] = c_scaled[:low_env.num_envs]
        for _ in range(self.cfg.nav.low_level_steps_per_high):
            low_obs = low_env.get_observations()
            with torch.no_grad():
                low_actions = self.low_level_policy(low_obs.detach())
            low_env.step(low_actions.detach())

        # 收集低层结果 → 构建高层观测
        self.compute_observations()
        self.compute_reward()

        clip_obs = 100.
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def compute_observations(self):
        # 从低层取本体感知
        low_env = self.low_level_env
        op = low_env.proprio_buf[:self.num_envs]
        hl_prop = torch.cat([op, self.commands], dim=1)  # 62维

        # 相对子目标位置 (世界系 → 机体系)
        robot_xy = low_env.root_states[:self.num_envs, :2]
        subgoal_world = self.current_subgoal
        delta_world = subgoal_world - robot_xy
        delta_body = quat_rotate_inverse(
            low_env.base_quat[:self.num_envs],
            torch.cat([delta_world, torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        )

        # 特权地图编码
        M3D, M2D = self._get_privileged_maps(robot_xy)
        z_m3d = self.map3d_encoder(M3D)
        z_m2d = self.map2d_encoder(M2D)
        map_code = torch.cat([z_m3d, z_m2d], dim=1)

        self.obs_buf = torch.cat([map_code, hl_prop, delta_body], dim=1)

    def compute_reward(self):
        low_env = self.low_level_env
        n = self.num_envs

        # 到达子目标
        robot_xy = low_env.root_states[:n, :2]
        dist_to_subgoal = torch.norm(self.current_subgoal - robot_xy, dim=1)
        self.subgoal_reached = dist_to_subgoal < self.cfg.nav.subgoal_arrival_threshold

        # 10 项奖励
        r_goal = (dist_to_subgoal < self.cfg.nav.goal_arrival_threshold).float() * self.reward_scales["goal_arrival"]
        r_ir = torch.tensor([self.state_explorer.get_intrinsic_reward(pos)
                             for pos in robot_xy.cpu().numpy()],
                           device=self.device) * self.reward_scales["state_count"]
        v_des = np.random.uniform(*self.cfg.nav.desired_speed_range)
        r_speed = torch.exp(-torch.abs(v_des - torch.norm(
            low_env.base_lin_vel[:n, :2], dim=1)) / 0.3) * self.reward_scales["desired_speed"]
        r_cmd_rate = -torch.sum(torch.square(self.commands - self.last_command), dim=1) * (-self.reward_scales["command_rate"])
        r_cmd_smooth = -torch.sum(torch.square(self.commands - 2*self.last_command + self.second_last_command), dim=1) * (-self.reward_scales["smooth_command"])
        r_track = -torch.norm(self.commands[:n, :2] - low_env.base_lin_vel[:n, :2], dim=1) * (-self.reward_scales["tracking_error"])
        r_body_vel = -(torch.square(low_env.base_lin_vel[:n, 2]).sum(dim=0) + torch.sum(torch.square(low_env.base_ang_vel[:n, :2]), dim=1)) * (-self.reward_scales["body_velocity"])
        r_posture = -torch.sum(torch.square(low_env.dof_pos[:n] - low_env.default_dof_pos), dim=1) * (-self.reward_scales["nominal_posture"])
        r_cmd_limit = (torch.abs(self.commands[:n]).sum(dim=1) > 5.0).float() * self.reward_scales["command_limit"]
        r_collision = self._detect_collision().float() * self.reward_scales["collision"]

        self.rew_buf = (r_goal + r_ir + r_speed + r_cmd_rate + r_cmd_smooth +
                        r_track + r_body_vel + r_posture + r_cmd_limit + r_collision)

        self.last_command = self.commands.clone()
        self.second_last_command = self.last_command.clone()

    def _detect_collision(self):
        low_env = self.low_level_env
        return torch.any(torch.norm(
            low_env.contact_forces[:self.num_envs, low_env.penalised_contact_indices, :],
            dim=-1) > 0.1, dim=1)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids] = 0.
        self.last_command[env_ids] = 0.
        self.second_last_command[env_ids] = 0.

    def _prepare_reward_function(self):
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        for key in list(self.reward_scales.keys()):
            if self.reward_scales[key] == 0:
                self.reward_scales.pop(key)

    def get_observations(self):
        return self.obs_buf

    def _get_privileged_maps(self, robot_xy):
        # 从预构建的地图缓存中查找每个 env 的局部地图
        # (简化: 返回占位符, 实际需在 build_privileged_maps 中实现)
        batch_M3D = torch.zeros(self.num_envs, 1, *self.cfg.nav.map_3d_size, device=self.device)
        batch_M2D = torch.zeros(self.num_envs, 1, *self.cfg.nav.map_2d5_size, device=self.device)
        return batch_M3D, batch_M2D
```

- [ ] **Step 2: 高层教师训练脚本**

```python
# legged_gym/scripts/train_high_teacher.py
import os
import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.hipan.terrain import WFCTerrain
from legged_gym.envs.hipan.pgcl import PGCLManager, AStarPlanner

def train_high_teacher(args):
    # 1. 加载低层学生 (已训练, 冻结)
    from legged_gym.scripts.train_low_student import load_low_student
    low_env, low_student_policy = load_low_student(args)

    # 2. 构建高层环境
    high_cfg, high_train_cfg = task_registry.get_cfgs(name="hipan_high_teacher")
    high_env, _ = task_registry.make_env(name="hipan_high_teacher", args=args, env_cfg=high_cfg)
    high_env.set_low_level(low_env, low_student_policy)

    # 3. 构建 PGCL + WFC 地形
    planner = AStarPlanner()
    for i in range(high_cfg.env.num_envs):
        pgcl = PGCLManager(initial_d=high_cfg.nav.pgcl_initial_d,
                           d_step=high_cfg.nav.pgcl_d_step,
                           path_planner=planner)
        high_env.pgcl_managers.append(pgcl)

    # 4. PPO 训练
    runner, train_cfg = task_registry.make_alg_runner(
        env=high_env, name="hipan_high_teacher", args=args, train_cfg=high_train_cfg)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                 init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    args.task = "hipan_high_teacher"
    train_high_teacher(args)
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/envs/hipan/high_level/high_level_teacher.py legged_gym/scripts/train_high_teacher.py
git commit -m "feat: 高层教师 HighLevelTeacher (双地图3D/2D编码 + 内嵌冻结低层学生 + PGCL + Intrinsic Reward)"
```

---

### Task 11: 高层学生 HighLevelStudent

**Files:**
- Create: `legged_gym/envs/hipan/high_level/high_level_student.py`
- Create: `legged_gym/scripts/train_high_student.py`

- [ ] **Step 1: 创建高层学生 — 深度图替代特权地图**

```python
# legged_gym/envs/hipan/high_level/high_level_student.py
import torch
from legged_gym.envs.hipan.high_level.high_level_teacher import HighLevelTeacher

class HighLevelStudent(HighLevelTeacher):
    """HiPAN 高层学生策略 πH_S: DAgger蒸馏, 深度图 → 5D导航指令"""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.gru_hidden = None

    def _init_buffers(self):
        super()._init_buffers()
        # 深度图编码器: CNN + GRU
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
        ).to(self.device)

        # GRU 记忆单元
        self.gru = torch.nn.GRU(input_size=128, hidden_size=128,
                                num_layers=1, batch_first=False).to(self.device)
        self.gru_hidden = torch.zeros(1, self.num_envs, 128, device=self.device)

        # 将隐向量映射到感知空间 z_s (与教师地图编码同维度)
        self.latent_projector = torch.nn.Linear(128, 32).to(self.device)

        # 学生主干 (替代教师 backbone)
        student_input_dim = 32 + 62 + 3  # z_s + op(62) + goal(3)
        self.backbone_student = torch.nn.Sequential(
            torch.nn.Linear(student_input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 5),
        ).to(self.device)

    def compute_observations(self):
        low_env = self.low_level_env
        op = low_env.proprio_buf[:self.num_envs]
        hl_prop = torch.cat([op, self.commands], dim=1)

        # 相对目标位置
        robot_xy = low_env.root_states[:self.num_envs, :2]
        goal_world = self.global_goal
        delta_world = goal_world - robot_xy
        from isaacgym.torch_utils import quat_rotate_inverse
        delta_body = quat_rotate_inverse(
            low_env.base_quat[:self.num_envs],
            torch.cat([delta_world, torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        )

        # 深度图 → CNN → GRU → z_s
        depth_img = self._get_depth_images()  # (N, 1, H, W)
        depth_feat = self.depth_encoder(depth_img)  # (N, 128)
        depth_feat_seq = depth_feat.unsqueeze(0)    # (1, N, 128) for GRU
        gru_out, self.gru_hidden = self.gru(depth_feat_seq, self.gru_hidden)
        z_s = self.latent_projector(gru_out.squeeze(0))

        self.obs_buf = torch.cat([z_s, hl_prop, delta_body], dim=1)

    def _get_depth_images(self):
        """从仿真相机获取深度图 (简化: Isaac Gym 相机 API 在后续实现)"""
        return torch.zeros(self.num_envs, 1, 180, 320, device=self.device)
```

- [ ] **Step 2: 高层学生 DAgger 训练脚本**

```python
# legged_gym/scripts/train_high_student.py
import os
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.dagger import DAggerTrainer

def train_high_student(args):
    # 1. 加载已训练高层教师 + 低层学生
    teacher_cfg, teacher_train_cfg = task_registry.get_cfgs(name="hipan_high_teacher")
    teacher_cfg.env.num_envs = 128
    teacher_env, _ = task_registry.make_env(name="hipan_high_teacher", args=args, env_cfg=teacher_cfg)
    teacher_train_cfg.runner.resume = True
    teacher_runner, _ = task_registry.make_alg_runner(
        env=teacher_env, name="hipan_high_teacher", args=args, train_cfg=teacher_train_cfg)
    teacher_policy = teacher_runner.get_inference_policy(device=teacher_env.device)

    # 2. 创建学生
    from legged_gym.envs.hipan.high_level.high_level_student import HighLevelStudent
    student_cfg, _ = task_registry.get_cfgs(name="hipan_high_student")
    student_cfg.env.num_envs = 128
    student_env, _ = task_registry.make_env(name="hipan_high_student", args=args, env_cfg=student_cfg)
    student_env.set_low_level(teacher_env.low_level_env, teacher_env.low_level_policy)

    # 3. DAgger
    optimizer = torch.optim.Adam(
        list(student_env.depth_encoder.parameters()) +
        list(student_env.gru.parameters()) +
        list(student_env.latent_projector.parameters()) +
        list(student_env.backbone_student.parameters()),
        lr=1e-4
    )

    dagger = DAggerTrainer(teacher_policy, student_env.backbone_student,
                           optimizer, student_env.device)
    for iteration in range(15):
        loss = dagger.train(student_env,
                            lambda obs: student_env.forward(obs),
                            lambda obs: teacher_policy(obs),  # 教师标注
                            num_iterations=1, steps_per_iter=100)
        print(f"DAgger iter {iteration}: loss={loss:.6f}")

    # 保存
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'logs', 'hipan_high_student', 'student.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'depth_encoder': student_env.depth_encoder.state_dict(),
        'gru': student_env.gru.state_dict(),
        'latent_projector': student_env.latent_projector.state_dict(),
        'backbone': student_env.backbone_student.state_dict(),
    }, save_path)

if __name__ == '__main__':
    args = get_args()
    args.task = "hipan_high_student"
    train_high_student(args)
```

- [ ] **Step 3: Commit**

```bash
git add legged_gym/envs/hipan/high_level/high_level_student.py legged_gym/scripts/train_high_student.py
git commit -m "feat: 高层学生 HighLevelStudent (深度图CNN+GRU替代特权地图 + DAgger蒸馏)"
```

---

### Task 12: 全流程验证脚本

**Files:**
- Create: `legged_gym/tests/test_hipan.py`

- [ ] **Step 1: 阶段性验证测试**

```python
# legged_gym/tests/test_hipan.py
import torch
from legged_gym.utils import get_args, task_registry

def test_low_teacher_env():
    """验证低层教师环境能正确初始化和 step"""
    args = get_args()
    args.task = "hipan_low_teacher"
    args.headless = True
    args.num_envs = 4

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 4
    env_cfg.terrain.mesh_type = 'plane'  # 简化测试用平地
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obs = env.get_observations()
    assert obs.shape == (4, env_cfg.env.num_observations), \
        f"Obs shape mismatch: {obs.shape} vs {(4, env_cfg.env.num_observations)}"

    for _ in range(100):
        actions = torch.randn(4, 12, device=env.device)
        obs, priv, rew, done, info = env.step(actions)
        assert obs.shape == (4, env_cfg.env.num_observations)
        assert rew.shape == (4,)

    print("PASS: Low-level teacher environment works")

if __name__ == '__main__':
    test_low_teacher_env()
```

- [ ] **Step 2: Commit**

```bash
git add legged_gym/tests/test_hipan.py
git commit -m "test: HiPAN 低层教师环境冒烟测试"
```

---

### 验证方案

| 阶段 | 验证内容 | 方法 |
|------|---------|------|
| 低层教师 | 5D指令跟踪误差 < 5% | 训练后 rollout, 统计各维度 RMSE |
| 低层学生 | 动作与教师偏差 < 10% | DAgger后同环境对比输出 MSE |
| 高层教师 | WFC环境 SR > 90%, SPL > 80 | 300个随机 start-goal 对评估 |
| 高层学生 | 与教师 SR/SPL 偏差 < 10% | 同评估集对比 |
