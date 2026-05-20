# HiPAN Low-Level Teacher 训练复盘

## 6 个阶段

| 阶段 | 日期 | 关键事件 |
|---|---|---|
| 1. Go2 步态竞赛 | 4/28-30 | 建立 gait 奖励、速度惩罚体系 |
| 2. HiPAN 架构搭建 | 5/17 | 四阶段管线、WFC地形、PGCL、DAgger |
| 3. 加法 Reward 时代 | 5/17-18 | 机器狗站着不动，首次训练失败 |
| 4. 乘法 Reward | 5/19 | 采用 WTW 结构 `R=(R_pos+αR_en)×exp(R_neg/σ)` |
| 5. Bug 猎杀 | 5/19-20 | body_roll π偏移、pitch缺失、abs改squared、domain_encoder未训练 |
| 6. σ 与权重调优 | 5/20 | vel/yaw/height σ 收紧、gait 奖励开启、height 指令对齐站姿 |

---

## 12 个错误

1. **`only_positive_rewards=True`** — 掩盖负奖励问题而非修复根因
2. **乘法 aux 权重过大** — `exp(-R_aux) → 0`，全部 reward 归零，策略随机行动
3. **`sigma_rew_neg=0.02`** — 门控太紧，训练早期熵爆炸（std 从 1.0 飙至 14.35）
4. **`body_roll = atan2(g_x, g_z)`** — 直立机器狗得到 π rad，cmd_roll ≈ 0，error = π，reward ≈ 0
5. **`body_orientation = |g_y|`** — 只惩罚 roll 倾斜，遗漏 pitch 分量（g_x）
6. **用 `abs` 而非 `squared` 误差** — 梯度特性不一致，未对齐 WTW
7. **`en_alpha=1.0`** — 静止时 CoT=0 → R_en=1.0（满分），策略学会"不动以节能"
8. **`dof_pos=-0.05`** 太弱 — 策略可自由偏离默认姿态至匍匐
9. **gait 奖励全关** — 策略无迈步动机，静态姿态是最优解
10. **height 指令 `[0.35, 0.40]` 高于默认站姿** — 机器人需主动伸腿才能追上，策略选择不费劲
11. **未经同意提议改 `action_scale`** — 违反用户设置的原则
12. **读文件不仔细** — 漏掉用户对 height 指令的编辑，报告了过时值

---

## 12 条启示

1. **静态匍匐是局部最优** — 如果 R_pos(匍匐) + R_en(最大) > R_pos(行走) + R_en(较低)，策略会理性选择不动。必须让不动无利可图
2. **零动作测试是金标准** — 证明 PD + 默认姿态可独立站立（base_z ≈ 0.29m）。若训练后策略倒塌，是主动选择的结果
3. **Train/Play 一致** — 两者仅 action 来源不同（sample vs deterministic mean），已验证行为相同
4. **WTW 不用 gait 奖励是因为策略看得见步态时钟** — 我们的观测缺时钟信号 → 需要显式 gait 奖励
5. **Squared > Abs** — 完美追踪处梯度为零（平滑峰值），中等误差处梯度更强
6. **DT 缩放使每步值极小** — `sigma_rew_neg` 必须根据累积 dt 缩放后的惩罚值来校准，不能看 raw 值
7. **height tracking 主导 R_pos** — 权重 1.0-3.0 且信号强，是主要的优化驱动力
8. **dof_pos 是最强 R_neg 项** — 在 -0.2 下贡献约 40% 的 R_neg，但面对 `sigma_rew_neg=0.04` 仍然不够
9. **用户偏好 reward shaping 而非硬约束** — 不加终止条件、不缩 action_scale、不改观测空间
10. **一次只改一个维度** — 用户明确要求这个纪律
11. **会话开始检查 git diff 和配置文件** — 用户可能在会话间做了编辑
12. **Go2 坐标系**: x-前 y-左 z-上, roll = `atan2(g_y, -g_z)`, pitch = `atan2(g_x, -g_z)`

---

## 当前配置状态

```python
# 指令范围 — 匹配默认站姿 ~0.29m
height = [0.27, 0.30]

# Tracking sigma — 全面收紧
tracking_sigma_vel    = 0.05
tracking_sigma_yaw    = 0.1
tracking_sigma_height = 0.03
tracking_sigma_roll   = 0.1

# 能量正则化
en_alpha  = 0.2
en_sigma  = 1000.0
en_eps    = 0.05

# R_neg 门控
sigma_rew_neg = 0.04

# R_pos 权重
velocity_tracking = 3.0
yaw_tracking      = 0.4
height_tracking   = 1.0
roll_tracking     = 0.5
feet_air_time     = 3.0

# R_neg 权重
body_orientation = -1.5
body_velocity    = -0.2
collision        = -1.0
action_rate      = -0.02
smooth_action    = -0.01
smooth_joint_vel = -0.002
smooth_joint_acc = -0.0000005
torque_usage     = -0.0005
joint_limit      = -0.5
dof_pos          = -0.2
gait_phase       = -0.1
```
