import torch
from torch.utils.data import Dataset


class DADataset(Dataset):
    """DAgger dataset: stores (observation, teacher_action, teacher_latent) triples."""

    def __init__(self, max_size=100000, latent_dim=32):
        self.observations = []
        self.teacher_actions = []
        self.teacher_latents = []
        self.max_size = max_size
        self.latent_dim = latent_dim

    def _placeholder(self):
        return torch.zeros(self.latent_dim)

    def add(self, obs, action, latent=None):
        if len(self.observations) >= self.max_size:
            idx = torch.randint(0, len(self.observations), (1,)).item()
            self.observations[idx] = obs.cpu()
            self.teacher_actions[idx] = action.cpu()
            self.teacher_latents[idx] = latent.cpu() if latent is not None else self._placeholder()
        else:
            self.observations.append(obs.cpu())
            self.teacher_actions.append(action.cpu())
            self.teacher_latents.append(latent.cpu() if latent is not None else self._placeholder())

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.teacher_actions[idx], self.teacher_latents[idx]


class DAggerTrainer:
    """DAgger online distillation trainer: alternates collection (student rollout + teacher
    labeling) and supervised training on the aggregated dataset."""

    def __init__(self, student_policy, optimizer, device,
                 latent_dim=32, batch_size=256, dagger_epochs=5):
        self.student = student_policy
        self.optimizer = optimizer
        self.device = device
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.dagger_epochs = dagger_epochs
        self.dataset = DADataset(latent_dim=latent_dim)

    def collect_and_label(self, env, student_policy_fn, teacher_inference_fn,
                          num_steps_per_env=24):
        obs = env.get_observations()
        for _ in range(num_steps_per_env):
            with torch.no_grad():
                if isinstance(obs, tuple):
                    student_out, _ = student_policy_fn(obs)
                else:
                    student_out = student_policy_fn(obs)

                teacher_out, teacher_latent = teacher_inference_fn(obs)

            self.dataset.add(
                obs.clone(),
                teacher_out.clone(),
                teacher_latent.clone() if teacher_latent is not None else None,
            )

            if isinstance(student_out, tuple):
                student_out = student_out[0]
            obs, _, _, _, _ = env.step(student_out)
            if isinstance(obs, tuple):
                obs = obs[0]

    def train_epoch(self):
        if len(self.dataset) == 0:
            return 0.0

        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True)
        total_loss = 0.0
        for batch_obs, batch_act, batch_lat in dataloader:
            batch_obs = [b.to(self.device) for b in batch_obs] if isinstance(
                batch_obs, list) else batch_obs.to(self.device)
            batch_act = batch_act.to(self.device)
            batch_lat = batch_lat.to(self.device)

            self.optimizer.zero_grad()
            pred_act, pred_lat = self.student(batch_obs)

            loss = torch.nn.functional.mse_loss(pred_act, batch_act)
            if batch_lat.abs().sum() > 0:
                loss = loss + torch.nn.functional.mse_loss(pred_lat, batch_lat)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, env, student_policy_fn, teacher_inference_fn,
              num_iterations=10, steps_per_iter=24):
        for iteration in range(num_iterations):
            self.collect_and_label(env, student_policy_fn,
                                   teacher_inference_fn, steps_per_iter)
            for _ in range(self.dagger_epochs):
                epoch_loss = self.train_epoch()
        return epoch_loss
