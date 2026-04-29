import numpy as np
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2RoughCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 235
        symmetric = True  #True :  set num_privileged_obs = None;    false: num_privileged_obs = observations + 187 ,set "terrain.measure_heights" to true
        num_privileged_obs = 235#num_observations + 187 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 65 # episode length in seconds
        waypoint_threshold = 1.0
        target_waypoints =[
            [8.0, 2.0, 0.3],
            [12.0, 3.0, 0.3],
            [17.0, 1.0, -0.1],
            [20.0, 6.0, -0.45],
            [23.0, 5.0, 0.2],
            [25.0, 3.0, 0.3],
            [30.0, 3.0, 0.3],
            [35.0, 3.0, 0.3],
            [40.0, 3.0, 0.3],
            [45.0, 6.0, 0.3],
            [50.0, 9.0, 0.3],
            [55.0, 3.0, 0.3],
            [60.0, 6.0, 0.6],
            [63.0, 3.0, 1.8],
            [66.0, 9.0, 2.6],
            [70.0, 6.0, 4.0],
            [72.0, 9.0, 4.5],
            [76.0, 6.0, 3.5],
            [80.0, 6.0, 2.1],
            [83.0, 6.0, 1.2],
            
        ]

    class terrain( LeggedRobotCfg.env ):
        mesh_type = 'competition' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.25 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 12.
        terrain_width = 12.
        num_rows= 9 # number of terrain rows (levels)
        num_cols = 1 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0,1, 0, 0, 0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
    class commands( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.5, 1.5] # min max [m/s] 
            lin_vel_y = [0, 0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20}  # [N*m/rad] 关节硬度
        damping = {'joint': 0.5}     # [N*m*s/rad]关节阻尼
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True
    class domain_rand:
        randomize_friction = False
        friction_range = [0.2, 1.5]
        randomize_base_mass = False
        added_mass_range = [-4., 4.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

        randomize_base_com = False
        added_com_range = [-0.15, 0.15]

        randomize_motor = False
        motor_strength_range = [0.8, 1.2]

    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.
            tracking_lin_vel = 0.0 #线速度
            tracking_ang_vel = -0.2
            tracking_goal_vel= 1.0
            lin_vel_z = -0. #Z轴速度
            ang_vel_xy = -0.02 #躯干角速度
            orientation = -0.2 #躯干水平度
            torques = -0.00005
            dof_vel = 0
            dof_acc = -1e-7
            dof_pos = -0.1
            base_height = -0.1 # 高度维持  
            feet_air_time =  0.2 #腾空时间
            collision = -1. #碰撞惩罚
            stumble = -0.2 #绊倒惩罚
            gait_phase = -0.2 # 步态相位惩罚权重  
            action_rate = -0.003 #动作平滑惩罚
            stand_still = -0.1
            dof_pos_limits =-0.01
            #goal_pos = 0.45

# step 1 
# negtive reward -> -0.001



# tracking_lin_vel 0.02
# dof_pos_limits -0.1   -10  -> -1 : -0.01 


# reward 100
# tracking_lin_vel 0.9

# base_height = -0.01 -> base_height = -0.05
# orientation =-0.0001  orientation= -0.1

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized
        gait_frequency = 1.8 # 步态频率(Hz)，Go2小跑通常在2.5~3.0之间
        # 相位偏移：假设脚的顺序是 [左前FL, 右前FR, 左后RL, 右后RR]
        gait_offsets = [0.0, 0.5, 0.5, 0.0] 
        gait_phase_scale = 0.5 # 占空比：0.5表示一半时间触地，一半时间腾空
class Go2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'

  
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 48 # per iteration
        max_iterations = 6000 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations

        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
