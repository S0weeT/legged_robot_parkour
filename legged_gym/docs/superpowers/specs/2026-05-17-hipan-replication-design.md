# HiPAN 仿真训练管线复刻 — 设计文档

## Context

在当前 Go2 端到端 RL 项目基础上，复刻 HiPAN 论文的四阶段仿真训练管线（不含真机部署），构建两层分层架构：高层导航策略 + 低层运动策略。所有代码改动位于新分支 `hipan`，原 `goal` 分支不动。

**重要约束**: 现有 `Go2Robot` 中的 waypoint 导航、command 劫持、目标点追踪等逻辑仅作为跑通框架的参考代码，不具有任何设计效力。HiPAN 模块从 `LeggedRobot`/`BaseTask` 直接继承，构建完全独立的逻辑链路，严格按照论文设计，不受现有导航代码影响。

## 目录与文件结构

```
legged_gym/envs/hipan/                    # 新建
├── __init__.py
├── low_level/
│   ├── low_level_config.py
│   ├── low_level_teacher.py              # πL_T: 继承 LeggedRobot
│   └── low_level_student.py              # πL_S: 继承 LeggedRobot
├── high_level/
│   ├── high_level_config.py
│   ├── high_level_teacher.py             # πH_T: 继承 BaseTask, 内嵌低层
│   └── high_level_student.py             # πH_S: 继承 BaseTask, 内嵌低层
├── terrain/
│   └── wfc_terrain.py                    # WFC 地形生成
└── pgcl.py                               # PGCL 课程管理器

legged_gym/utils/dagger.py                # 新建: DAgger 蒸馏工具
legged_gym/envs/__init__.py               # 修改: 注册 hipan 任务
legged_gym/envs/base/legged_robot.py      # 微调: 加少量钩子方法
```

继承关系: HiPAN 低层直接继承 `LeggedRobot`，高层直接继承 `BaseTask`。不继承 `Go2Robot`，不受现有 waypoint/command劫持逻辑任何影响。
改动原则: `legged_robot.py` 不改变现有行为，仅在需要处加少量钩子供 Hipan 子类复用。

## 两层的指令接口

```
c = [vx, vy, ωz, h, θx] ∈ R⁵

vx:  前向速度 (机体系)     ∈ [-1.5, 1.5] m/s
vy:  侧向速度 (机体系)     ∈ [-1.0, 1.0] m/s
ωz:  偏航角速度 (机体系)   ∈ [-1.5, 1.5] rad/s
h:   目标身高 (世界系)     ∈ [0.1, 0.4] m
θx:  目标侧倾角 (世界系)   ∈ [-1.0, 1.0] rad
```

## 阶段 1: 低层教师 πL_T

**继承**: `LeggedRobot`

**输入 (特权)**:

```
s = (op, c, xm, xd)

op (57维本体感知): q(12) + q̇(12) + p_F0~3(12) + I_0~3(4) + θ_B,xy(2) + ω_B(3) + a_{t-1}(12)
c  (5维): 导航指令
xm (5维特权): v_B(xyz) + h_B + θ^W_{B,x}
xd (特权域参数): 地形高度图 + 摩擦/质量/电机 → Encoder → z_d ∈ R³²
```

**动作**: a = Δq ∈ R¹², 经 PD 控制器: τ = Kp*(q₀ + Δq - q) - Kd*q̇

**奖励** (12项, 来自 HiPAN Table III):

| 奖励 | 权重 | 公式 |
|------|------|------|
| Velocity Tracking | +0.4 | exp(-‖[vx_c,vy_c] - v_xy‖²/0.25) |
| Yaw-Rate Tracking | +0.2 | exp(-|ωz_c - ωz|/0.25) |
| Height Tracking | +0.2 | exp(-|h_c - h|/0.0025) |
| Roll Tracking | +0.2 | exp(-|θx_c - θx|/0.05) |
| Action Rate | -0.01 | ‖a_t - a_{t-1}‖² |
| Smooth Action | -0.1 | ‖a_t - 2a_{t-1} + a_{t-2}‖² |
| Body Orientation | -0.5 | |θ_y| |
| Body Velocity | -0.2 | |vz|² + ‖ω_xy‖² |
| Smooth Joint | -0.001/-0.0001 | ‖q̇‖² + ‖q̈‖² |
| Torque Usage | -0.0001 | ‖τ‖² |
| Joint Limit | -10.0 | 1{超限} |
| Collision | -10.0 | 1{碰撞} |

**指令采样**: Grid-Adaptive Curriculum — 初始小范围，姿态跟踪达标后扩至全范围。

**训练**: PPO, 4096 envs, ~12h, 地形: stairs/holes/slopes/flat, 域随机化(摩擦/质量/电机/推力)。

## 阶段 2: 低层学生 πL_S

**输入 (板载)**:

```
o = [o⁵⁰_p, c]
o⁵⁰_p: 50步本体感知历史环缓冲区 (50×57)
c:     5D 导航指令
```

**估计器**:

```
e_d: o⁵⁰_p → ẑ_d ∈ R³²    (域参数隐向量，1D Conv + 池化)
e_m: o¹⁰_p → x̂_m ∈ R⁵      (运动状态，1D Conv + FC)
```

**网络**:

```
o⁵⁰_p → e_d → ẑ_d ─┐
o¹⁰_p → e_m → x̂_m ─┤
                    ├→ b_S → Δq (12维)
c ──────────────────┘
```

**DAgger 损失**: L = ‖a_T - a_S‖² + ‖z_d - ẑ_d‖² + ‖x_m - x̂_m‖²

**训练**: DAgger, 300 envs, ~2h, 冻结教师标注 → 学生优化 → 重采集迭代。

## 阶段 3: 高层教师 πH_T

**继承**: `BaseTask`, 内嵌已冻结的低层学生 πL_S 作为闭环执行器。

**输入 (特权)**:

```
s̄ = (x̄_M, ō_p, x_m, p^B_gs)

x̄_M: 双地图
  M_3D   ∈ R^(14×11×11)  3D体素占据图, 0.1m分辨率, ±0.5m局部范围
  M_2.5D ∈ R^(31×21)      2.5D高度图, 0.1m分辨率, ±1.0m局部范围

ō_p = [op, c_{t-1}] (62维)
x_m: 特权运动状态 (5维)
p^B_gs: 当前子目标相对机器人坐标 (3维)
```

**双地图**: M_3D 检测头顶障碍物(姿态调整), M_2.5D 感知前方结构(导航规划)。编码为 z_s ∈ R³²。

**网络**:

```
M_3D → 3D CNN ──┐
M_2.5D → 2D CNN ─┤--→ z_s → b_T → c ∈ R⁵
ō_p ──────────────┤
x_m ──────────────┤
p^B_gs ───────────┘
```

**高层步进**: 输出 c → 低层以 50Hz 执行 100ms(5步) → 收集返回状态 → 高层计算奖励。

**奖励** (10项, HiPAN Table II):

| 奖励 | 权重 | 公式 |
|------|------|------|
| Goal Arrival | +5.0 | 1{dist < 0.1m} |
| State Count (IR) | +0.5 | 1/√η(p_B,xy) |
| Desired Speed | +0.25 | exp(-|v_des - ‖v‖|/0.3) |
| Command Rate | -0.1 | ‖c_t - c_{t-1}‖² |
| Smooth Command | -0.1 | ‖c_t - 2c_{t-1} + c_{t-2}‖² |
| Tracking Error | -0.2 | ‖c - [v_B, ωz, h, θx]‖ |
| Body Velocity | -0.1 | |vz|² + ‖ω_xy‖² |
| Nominal Posture | -0.04 | ‖q - q₀‖² |
| Command Limit | -2.5 | 1{超限} |
| Collision | -2.5 | 1{碰撞} |

**Intrinsic Reward**: 离散网格(0.1m×0.1m)位置访问计数 η(p), r_IR = 1/√η。

**PGCL 课程**: d 从 1m 起步, 沿 A* 全局路径每 d 米放子目标。全部到达 → d += 1m → 升级。最终仅剩起始→终点。WFC 算法生成 6 个 20×20m 训练环境。

**训练**: PPO, 1024 envs, ~18h, start-goal 距离分三档 [5,10]/[10,20]/[20,30]m。

## 阶段 4: 高层学生 πH_S

**输入 (板载)**:

```
ō = (ō_D, ō_p, x̂_m, p^B_g)

ō_D:   深度图 180×320 (仿真虚拟深度相机)
ō_p:   [op, c_{t-1}] (62维)
x̂_m:   低层 e_m 输出 (5维)
p^B_g: 最终目标相对坐标 (3维)
```

**网络**:

```
ō_D → 2D CNN → GRU(128) → ẑ_s → b_S → ĉ ∈ R⁵
ō_p ───────────────────────────┤
x̂_m ───────────────────────────┤
p^B_g ──────────────────────────┘
```

**DAgger 损失**: L = ‖c - ĉ‖² + ‖z_s - ẑ_s‖²

**训练**: DAgger, 128 envs, ~6h, 教师标注感知隐向量+指令, 学生模仿。

## 总览

```
阶段1: 低层教师   PPO     4096envs  ~12h  → πL_T (需特权)
阶段2: 低层学生   DAgger   300envs   ~2h  → πL_S (本体历史+c)
阶段3: 高层教师   PPO     1024envs  ~18h  → πH_T (需地图+πL_S)
阶段4: 高层学生   DAgger   128envs   ~6h  → πH_S (深度图+πL_S)

训练顺序严格: 低层先收敛, 再嵌入高层。
```

## 验证方案

1. 阶段1: 低层教师在随机 5D 指令下各维度跟踪误差 < 5%
2. 阶段2: 低层学生跟踪精度与教师偏差 < 10%
3. 阶段3: 高层教师 WFC 环境 Success Rate > 90%, SPL > 80
4. 阶段4: 高层学生 Success Rate/SPL 与教师偏差 < 10%
