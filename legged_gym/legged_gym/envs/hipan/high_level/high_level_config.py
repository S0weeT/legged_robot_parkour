from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HighLevelCfg(LeggedRobotCfg):
    """HiPAN high-level navigation policy config: 10Hz, WFC terrain, PGCL curriculum, dual-map perception."""

    class env(LeggedRobotCfg.env):
        num_envs = 1024
        num_observations = 256
        num_actions = 5   # [vx, vy, wz, h, roll]
        episode_length_s = 120
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
        """Navigation parameters."""
        high_level_freq = 10   # Hz
        low_level_steps_per_high = 5  # 100ms = 5 * 0.02s
        desired_speed_range = [0.3, 1.2]  # m/s
        goal_arrival_threshold = 0.1       # m
        subgoal_arrival_threshold = 0.1    # m

        # Map parameters
        map_3d_resolution = 0.1
        map_3d_range = [-0.5, 0.5]
        map_3d_size = [14, 11, 11]
        map_2d5_resolution = 0.1
        map_2d5_range = [-1.0, 1.0]
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
        randomize_base_mass = False
        randomize_base_com = False
        randomize_motor = False
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
        num_steps_per_env = 100   # high-level 10Hz, 100 steps = 10s rollout
        max_iterations = 8000

        save_interval = 50
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
