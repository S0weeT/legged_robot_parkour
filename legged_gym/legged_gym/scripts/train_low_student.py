# legged_gym/scripts/train_low_student.py
"""HiPAN low-level student DAgger training script.

Loads a pre-trained LowLevelTeacher checkpoint as the labeling oracle, creates a
LowLevelStudent environment, and trains the student via DAgger online distillation.
"""
import os
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.dagger import DAggerTrainer


def load_low_teacher(args, num_envs=300):
    """Load a trained low-level teacher environment and its inference policy.

    Returns (teacher_env, teacher_policy_fn) where teacher_policy_fn maps
    observation -> actions_mean (deterministic PPO actor).
    """
    teacher_cfg, train_cfg = task_registry.get_cfgs(name="hipan_low_teacher")
    teacher_cfg.env.num_envs = num_envs
    teacher_env, _ = task_registry.make_env(
        name="hipan_low_teacher", args=args, env_cfg=teacher_cfg,
    )
    train_cfg.runner.resume = True
    teacher_runner, train_cfg = task_registry.make_alg_runner(
        env=teacher_env, name="hipan_low_teacher", args=args, train_cfg=train_cfg,
    )
    teacher_policy = teacher_runner.get_inference_policy(device=teacher_env.device)
    return teacher_env, teacher_policy


def train_low_student(args):
    # 1. Load trained teacher (labeling oracle)
    print("Loading teacher checkpoint ...")
    teacher_env, teacher_policy = load_low_teacher(args, num_envs=300)

    # 2. Create student environment
    print("Creating student environment ...")
    student_cfg, _ = task_registry.get_cfgs(name="hipan_low_student")
    student_cfg.env.num_envs = 300
    student_env, _ = task_registry.make_env(
        name="hipan_low_student", args=args, env_cfg=student_cfg,
    )

    # 3. Build DAgger optimizer over student's trainable networks
    optimizer = torch.optim.Adam(
        list(student_env.estimator_domain.parameters()) +
        list(student_env.estimator_motion.parameters()) +
        list(student_env.backbone_student.parameters()),
        lr=1e-4,
    )

    dagger = DAggerTrainer(
        student_policy=lambda obs: student_env.forward(obs),
        optimizer=optimizer,
        device=student_env.device,
    )

    # 4. Teacher labeling function: computes privileged observation from env state,
    #    runs teacher PPO inference, returns (action, domain_latent zd).
    def teacher_label_fn(_obs_buf):
        with torch.no_grad():
            # Compute privileged motion state xm (same as teacher's compute_observations)
            base_height = torch.mean(
                student_env.root_states[:, 2].unsqueeze(1) - student_env.measured_heights,
                dim=1,
            ).unsqueeze(1)
            body_roll = torch.atan2(
                student_env.projected_gravity[:, 0],
                student_env.projected_gravity[:, 2],
            ).unsqueeze(1)
            xm = torch.cat((
                student_env.base_lin_vel,   # v_B (3)
                base_height,                 # h_B (1)
                body_roll,                   # theta_x (1)
            ), dim=-1)

            # Compute privileged domain latent zd from domain parameters
            domain_input = student_env._get_domain_params()
            zd = student_env.domain_encoder(domain_input)

            # Build teacher's full privileged observation
            teacher_obs = torch.cat((
                student_env.proprio_buf,
                student_env.commands * student_env.commands_scale,
                xm,
                zd,
            ), dim=-1)

            action = teacher_policy(teacher_obs)
        return action, zd

    # 5. Student policy function (wraps env.forward for DAgger)
    def student_policy_fn(obs_buf):
        return student_env.forward(obs_buf)

    # 6. DAgger training loop
    print("Starting DAgger training ...")
    for iteration in range(20):
        dagger.collect_and_label(
            student_env, student_policy_fn, teacher_label_fn, num_steps_per_env=48,
        )
        loss = dagger.train_epoch()
        print(f"DAgger iter {iteration}: loss={loss:.6f}")

    # 7. Save student checkpoint
    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'hipan_low_student')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'student_checkpoint.pt')
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
