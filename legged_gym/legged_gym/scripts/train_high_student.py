# legged_gym/scripts/train_high_student.py
"""HiPAN high-level student DAgger training script.

Workflow:
  1. Load a trained low-level teacher checkpoint as the frozen low-level policy.
  2. Load a trained HighLevelTeacher checkpoint as the labeling oracle.
  3. Create a HighLevelStudent environment with depth + GRU perception.
  4. DAgger online distillation: student rollout, teacher labeling, supervised training.
  5. Save distilled student checkpoint (depth_encoder, GRU, latent_projector, backbone).

Usage:
  python legged_gym/scripts/train_high_student.py --task=hipan_high_student [--headless] [--num_envs=128]
"""

import os
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.dagger import DAggerTrainer


def train_high_student(args):
    # ------------------------------------------------------------------
    # 1. Load low-level student env + policy (frozen locomotion module)
    # ------------------------------------------------------------------
    print("Loading low-level student ...")
    from legged_gym.scripts.train_low_student import load_low_teacher

    low_env, low_policy = load_low_teacher(args, num_envs=128)

    # ------------------------------------------------------------------
    # 2. Load high-level teacher env + policy (labeling oracle)
    # ------------------------------------------------------------------
    print("Loading high-level teacher ...")
    teacher_cfg, teacher_train_cfg = task_registry.get_cfgs(name="hipan_high_teacher")
    teacher_cfg.env.num_envs = 128
    teacher_env, _ = task_registry.make_env(
        name="hipan_high_teacher", args=args, env_cfg=teacher_cfg,
    )
    teacher_env.set_low_level(low_env, low_policy)
    teacher_train_cfg.runner.resume = True
    teacher_runner, _ = task_registry.make_alg_runner(
        env=teacher_env, name="hipan_high_teacher", args=args,
        train_cfg=teacher_train_cfg,
    )
    teacher_policy = teacher_runner.get_inference_policy(device=teacher_env.device)

    # ------------------------------------------------------------------
    # 3. Create student environment
    # ------------------------------------------------------------------
    print("Creating high-level student ...")
    student_cfg, _ = task_registry.get_cfgs(name="hipan_high_student")
    student_cfg.env.num_envs = 128
    student_env, _ = task_registry.make_env(
        name="hipan_high_student", args=args, env_cfg=student_cfg,
    )
    student_env.set_low_level(low_env, low_policy)

    # ------------------------------------------------------------------
    # 4. DAgger setup
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        list(student_env.depth_encoder.parameters()) +
        list(student_env.gru.parameters()) +
        list(student_env.latent_projector.parameters()) +
        list(student_env.backbone_student.parameters()),
        lr=1e-4,
    )

    dagger = DAggerTrainer(
        student_policy=lambda obs: student_env.forward(obs),
        optimizer=optimizer,
        device=student_env.device,
    )

    # ------------------------------------------------------------------
    # 5. Teacher labeling function
    # ------------------------------------------------------------------
    def teacher_label_fn(obs_buf):
        with torch.no_grad():
            action = teacher_policy(obs_buf)
        return action, student_env.map_code if hasattr(student_env, 'map_code') else torch.zeros(
            student_env.num_envs, 32, device=student_env.device)

    def student_policy_fn(obs_buf):
        return student_env.forward(obs_buf)

    # ------------------------------------------------------------------
    # 6. DAgger training
    # ------------------------------------------------------------------
    print("Starting DAgger training ...")
    for iteration in range(15):
        dagger.collect_and_label(student_env, student_policy_fn, teacher_label_fn, 100)
        loss = dagger.train_epoch()
        print(f"DAgger iter {iteration}: loss={loss:.6f}")

    # ------------------------------------------------------------------
    # 7. Save student checkpoint
    # ------------------------------------------------------------------
    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'hipan_high_student')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'student.pt')
    torch.save({
        'depth_encoder': student_env.depth_encoder.state_dict(),
        'gru': student_env.gru.state_dict(),
        'latent_projector': student_env.latent_projector.state_dict(),
        'backbone': student_env.backbone_student.state_dict(),
    }, save_path)
    print(f"Student saved to {save_path}")


if __name__ == '__main__':
    args = get_args()
    args.task = "hipan_high_student"
    train_high_student(args)
