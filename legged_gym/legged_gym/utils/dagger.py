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
                self.teacher_latents[idx] = torch.zeros(1)
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
