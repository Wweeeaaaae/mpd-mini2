# MPD Principles Notes

这份说明只整理项目的原理性理解，不记录代码改动过程。重点回答四件事：

- 这套方法整体在做什么
- 数据为什么按现在这种方式生成
- GPMP2 和推理时 guidance 数学上怎样起作用
- 当前结果能说明什么泛化性，不能说明什么

## 1. 整体方法

这个项目不是“神经网络直接输出路径”，而是两阶段方法：

1. 用传统规划器生成训练轨迹  
   当前数据生成链路是 `RRTConnect -> Hybrid smoothing -> GPMP2`，最后把 free trajectory 存下来作为训练数据。

2. 用扩散模型学习轨迹先验  
   训练时，模型学习的是：

```text
p_theta(X | q_s, q_g)
```

其中：

- `X` 是整条轨迹
- `q_s` 是起点
- `q_g` 是终点

因此模型学到的核心不是“环境地图到轨迹”的直接映射，而是“在这类任务分布下，起终点给定时，合理轨迹通常长什么样”。

推理时再把当前环境代价加进来，所以更准确的理解是：

```text
p(X | q_s, q_g, E)  proportional to  p_theta(X | q_s, q_g) * exp(-J_E(X))
```

这里：

- `E` 是当前环境
- `J_E(X)` 是碰撞和平滑等代价

也就是说，扩散模型提供 prior，guidance 负责把 prior 拉向当前环境下更可行的轨迹。

## 2. 数据生成、多样性与环境设置

当前每个数据集子目录对应一个固定环境。

- Panda 3D 使用固定的 `EnvSpheres3D`
- Mini2 当前使用固定的 `EnvSpheres3DMini2`

这些环境里的障碍物中心和半径都是直接写死在环境类里的，所以同一个数据集内部，障碍物布局不变。

当前数据多样性主要来自两部分：

1. 起点和终点随机采样  
   先在关节空间均匀采样，再筛掉碰撞配置，因此不同 context 的 start/goal 不同。

2. 同一任务下的轨迹随机性  
   传统规划器本身有随机性，同一个 start/goal 可以得到多条不同候选轨迹。

所以当前数据集的多样性是：

- task diversity：不同起终点
- trajectory diversity：同一任务下不同可行轨迹

不是：

- environment diversity：不同障碍物布局

原作者的 Panda 数据集也是同样思路，即固定环境上训练。

这样做在当前模型里是自洽的，因为训练条件里只有起点和终点，没有障碍物编码。  
如果直接把很多不同障碍物布局混到同一个训练集里，而模型条件又不包含环境信息，那么同样的 start/goal 会对应很多互相冲突的轨迹分布，训练会更难稳定。

## 3. GPMP2 在数据生成阶段做什么

GPMP2 的角色不是从零搜路径，而是对已有轨迹做局部优化。

可以把轨迹写成：

```text
X = {x_0, x_1, ..., x_(H-1)}
x_t = [q_t; qdot_t]
```

其中：

- `q_t` 是关节位置
- `qdot_t` 是关节速度

当前优化目标可以概括成：

```text
J(X) = J_start + J_goal + J_collision + J_gp
```

其中：

- `J_start`：起点约束
- `J_goal`：终点约束
- `J_collision`：碰撞代价
- `J_gp`：平滑项，也就是 GP prior

平滑项采用常速度模型：

```text
x_(t+1) ~= Phi * x_t

Phi = [ I      dt*I
        0      I   ]
```

所以每一段的误差是：

```text
e_t = x_(t+1) - Phi * x_t
```

总代价写成：

```text
J_gp(X) = sum over t=0..H-2 of  e_t^T * Q^(-1) * e_t
```

这会鼓励轨迹平滑、速度连续。

碰撞项可以近似理解为：

```text
J_collision(X) =
sum over t and l of  max(0, m_l - d(p_(t,l)(q_t)))
```

其中：

- `p_(t,l)(q_t)` 是机器人在时刻 `t` 的第 `l` 个碰撞检测点
- `d(.)` 是该点到障碍物的 signed distance
- `m_l` 是安全 margin

如果某个碰撞点离障碍物已经足够远，则这一项为 0；如果太近或穿进障碍物，这一项变大。

当前 Mini2 数据生成里出现 invalid trajectory 的核心原因是：

- 优化阶段主要考虑障碍物、自碰撞、边界和平滑
- joint limits 主要是在最后做过滤
- 因此会出现“对障碍物 free，但关节轻微超限”的轨迹

这也是为什么会看到：

- `collision_intensity = 0`
- 但最终 `free = 0`

因为这里的 invalid 不只包含障碍物碰撞，还包含 joint-limit 违规。

## 4. 推理时的 guidance 怎样工作

### 4.1 当前真正被优化的变量

推理时，扩散模型处理的不是单个关节角，而是一整条轨迹：

```text
X = {x_0, x_1, ..., x_(H-1)}
x_t = [q_t; qdot_t]
```

其中：

- `q_t` 是关节位置
- `qdot_t` 是关节速度

当前默认训练模型都使用 `include_velocity=True`，所以状态本身包含速度。  
以当前 Mini2 结果为例，`H = 64`，状态维度是 `12 = 6 个 q + 6 个 qdot`。  
当前推理还固定 `trajectory_duration = 5.0`，因此：

```text
dt = 5.0 / 64 = 0.078125
```

guide 的目标不是优化模型参数，而是直接对当前反向扩散中的轨迹样本 `X` 求梯度并更新它。

### 4.2 参与扩散的总代价

当前推理里真正参与 guidance 的总代价可以写成：

```text
J_total(X) = J_coll(X) + J_gp(X)
```

其中：

- `J_coll(X)` 是碰撞代价
- `J_gp(X)` 是平滑代价，也就是常速度 `Gaussian Process prior（高斯过程先验）`

在当前默认设置里，两者在真正写回轨迹时使用的梯度权重分别是：

- `weight_grad_cost_collision = 1e-2`
- `weight_grad_cost_smoothness = 1e-7`

这意味着当前配置明显是“碰撞优先，平滑次之”。  
平滑项更像一个很弱的正则，而不是主导优化目标。

### 4.3 碰撞代价怎样定义

碰撞代价不是直接拿关节角和障碍物比较，而是先用前向运动学把每个 `q_t` 映射成一组机器人碰撞点在空间中的位置。  
在当前默认 `collision spheres（碰撞球）` 近似下，这些点可以直接理解成：

- 机器人每个碰撞球中心在时刻 `t` 的三维坐标

记第 `t` 个时间点第 `i` 个碰撞点的位置为：

```text
p_(t,i)
```

对于某个障碍物 `o`，先计算这个点到障碍物表面的 `signed distance（带符号距离）`：

```text
d_(t,i,o) = sdf_o(p_(t,i))
```

这里：

- `d > 0` 表示点在障碍物外部
- `d = 0` 表示点在障碍物表面
- `d < 0` 表示点已经进入障碍物内部

如果障碍物是球，`signed distance（带符号距离）` 就是：

```text
d_(t,i,o) = ||p_(t,i) - c_o|| - r_o
```

其中：

- `c_o` 是球心
- `r_o` 是球半径

但用于优化的不是距离本身，而是一个 `penalty（惩罚项）`。  
当前 object collision（物体碰撞）项使用的形式可以概括为：

```text
phi_(t,i,o) = max(0, m_i + delta - d_(t,i,o))
```

其中：

- `m_i` 是第 `i` 个碰撞点自身的 `collision margin（碰撞裕量）`
- `delta` 是全局 `cutoff margin（额外安全裕量）`

当前推理里：

- Panda 默认 `obstacle_cutoff_margin = 0.05`
- Mini2 默认 `obstacle_cutoff_margin = 0.05`

所以这里的物理意义是：

- 如果碰撞点离障碍物表面的距离大于“自身碰撞球半径 + 5 cm”，这一项为 0
- 如果虽然还没撞上，但已经进入这个安全缓冲区，这一项开始线性增大
- 如果已经穿障，代价继续增大

因此 guidance 并不是只在真正碰撞时才起作用，而是在“离障碍物太近”时就开始把轨迹往外推。

对于同一个碰撞点，环境里可能有多个障碍物。当前聚合方式更接近：

```text
phi_(t,i) = max over o of phi_(t,i,o)
```

也就是只看该碰撞点在当前时刻最危险的那个障碍物。  
再把所有碰撞点的惩罚求和，得到该时间点的障碍物误差：

```text
e_t^(obj) = sum over i of phi_(t,i)
```

整条轨迹的物体碰撞代价就是：

```text
J_obj(X) = sum over t=1..H-1 of e_t^(obj)
```

这里从 `t=1` 开始，不包含起点。当前 `sigma_coll = 1.0`，因此这条支路没有再被额外放缩。

当前默认 `use_guide_on_extra_objects_only=False` 时，碰撞总代价不是只含一项，而是：

```text
J_coll(X) = J_self(X) + J_obj(X) + J_ws(X)
```

其中：

- `J_self(X)` 是 `self-collision（自碰撞）`
- `J_obj(X)` 是 `object collision（物体碰撞）`
- `J_ws(X)` 是 `workspace boundary collision（工作空间边界碰撞）`

如果显式打开 `use_guide_on_extra_objects_only=True`，才会只用额外障碍物那一支做碰撞引导。

自碰撞项和工作空间边界项在思想上与物体碰撞相同，都是“先算离危险边界还有多远，再把距离变成 ReLU 型惩罚”。  
区别只在于底层距离量不同：

- 物体碰撞使用 `signed distance（带符号距离）`
- 自碰撞使用机器人若干碰撞点对之间的欧氏距离
- 边界碰撞使用碰撞点到工作空间六个边界面的距离

### 4.4 平滑代价怎样定义

当前参与扩散的平滑项不是终端里打印的 `cost smoothness`，而是常速度 `Gaussian Process prior（高斯过程先验）` 的二次型代价。

先定义状态转移：

```text
x_(t+1) ~= Phi * x_t

Phi = [ I      dt*I
        0      I   ]
```

当前 `dt = 0.078125`，所以这个模型表达的是：

- 位置按当前速度前进一步
- 速度在局部保持不变

于是每一段的误差是：

```text
e_t^(gp) = x_(t+1) - Phi * x_t
```

把它展开后就是：

```text
e_t^(gp) =
[ q_(t+1) - q_t - dt*qdot_t ;
  qdot_(t+1) - qdot_t ]
```

所以它同时惩罚两类不合理行为：

- 相邻两点的位置变化不符合当前速度
- 相邻两点的速度变化过大

整条轨迹的平滑代价为：

```text
J_gp(X) = sum over t=0..H-2 of  (e_t^(gp))^T * Q^(-1) * e_t^(gp)
```

当前 `sigma_gp = 1.0`，因此过程噪声逆协方差可以看成单位尺度。  
这项的直观作用不是“让路径更短”，而是让轨迹更符合常速度运动学，更少出现突然加速、突然减速或局部尖拐。

### 4.5 guide 对哪个量求梯度

guide 真正求的是总代价对整条轨迹状态的梯度：

```text
grad_X J_total(X)
```

这里：

```text
X in R^(H x D)
```

也就是对每个时间点的 `[q_t; qdot_t]` 都求梯度。  
不是对网络参数求梯度，也不是对障碍物参数求梯度。

碰撞项对某个时间点关节状态 `q_t` 的梯度，按链式法则可以写成：

```text
grad_(q_t) J_coll =
sum over i,o of
    [d phi_(t,i,o) / d d_(t,i,o)]
    * [d d_(t,i,o) / d p_(t,i)]
    * [d p_(t,i) / d q_t]
```

这三项分别表示：

- 第一项：惩罚对距离的导数  
  当 `d_(t,i,o) < m_i + delta` 时导数为 `-1`，否则为 `0`

- 第二项：距离场对空间点位置的梯度  
  表示“朝哪个方向移动，这个碰撞点会离障碍物更远”

- 第三项：前向运动学雅可比  
  表示“关节变量怎样变化，会让该碰撞点在空间里怎样移动”

因此碰撞梯度的物理意义很直接：  
告诉每个关节“应朝哪个方向微调，才能让机器人身上的碰撞点更远离障碍物”。

平滑项对中间状态 `x_t` 的梯度来自两侧相邻误差：

```text
grad_(x_t) J_gp =
    2 * Q^(-1) * (x_t - Phi * x_(t-1))
    - 2 * Phi^T * Q^(-1) * (x_(t+1) - Phi * x_t)
```

它的作用是把当前点往“两边都更一致”的方向拉，因此会同时抑制：

- 速度突变
- 与相邻点不协调的局部折线

### 4.6 梯度怎样影响逆扩散

当前实现不是只在最后统一做一次优化，而是在每一步逆扩散的模型均值附近插入多次 guide 更新。

先有模型给出的当前去噪均值：

```text
xbar_(t-1)^(0) = mu_theta(x_t, t)
```

然后做 `K` 次 guidance：

```text
xbar_(t-1)^(k+1) =
    Pi_hard( xbar_(t-1)^(k) + g(xbar_(t-1)^(k)) )
```

其中：

- `g(.)` 是代价的负梯度方向
- `Pi_hard(.)` 表示重新施加起点和终点硬约束

当前实现还会对每一项梯度先做 `gradient clipping（梯度裁剪）`，默认是按范数裁剪，`max_grad_norm = 1.0`，再乘权重并相加。  
因此更准确地说，实际更新方向是：

```text
g(X) = -
[
  1e-2 * clip(grad_X J_self)
  + 1e-2 * clip(grad_X J_obj)
  + 1e-2 * clip(grad_X J_ws)
  + 1e-7 * clip(grad_X J_gp)
]
```

同时起点和终点梯度会被强制置零，所以 guidance 不会改动这两个硬约束点。

最后再加噪声：

```text
x_(t-1) = xbar_(t-1)^(K) + sigma_t * xi
```

当前默认设置下：

- `n_diffusion_steps = 25`
- `start_guide_steps_fraction = 0.25`
- `t_start_guide = ceil(25 * 0.25) = 7`
- `n_diffusion_steps_without_noise = 5`
- `n_guide_steps = 5`

因此 guide 会在最后 12 个反向更新上起作用，并且每个更新再做 5 次梯度修正，总体上约进行 60 次 cost-guided（代价引导）微调。  
当前还把额外采样噪声系数固定成 `0.5`，所以 guide 推出来的方向不会立刻被过强噪声冲掉。

这就是这套方法的核心：

- 扩散 prior 负责生成“像训练数据”的轨迹
- test-time guidance（测试时引导）负责把这条轨迹往“当前环境下更不撞、更符合常速度先验”的方向推

### 4.7 终端打印的评价指标怎样计算

终端里打印的指标不参与上面的 guide 更新，它们只是对最终结果的后验统计。

首先，最终一条轨迹是否算 `free（无碰撞）`，不是只看原始 64 个 support points，而是先做更细的时间插值，再做碰撞检查。  
当前评估时默认：

- `num_interpolation = 5`
- `margin = 0.01`

也就是说，最终 free / collision 判定比 guide 阶段的“提前留 5 cm 安全距离”更接近硬判定。

在这个判定基础上：

```text
success = 1   if 至少存在一条 free trajectory
success = 0   otherwise
```

```text
percentage free trajs =
    (# free trajectories) / (# total trajectories)
```

```text
percentage collision intensity =
    (# colliding interpolated waypoints) / (# all interpolated waypoints)
```

因此 `percentage collision intensity` 是按路径点统计，不是按轨迹条数统计。

对最终 free trajectories，终端里还会打印三个后验代价：

```text
cost path length_i =
    sum over t=0..H-2 of ||q_(t+1) - q_t||_2
```

```text
cost smoothness_i =
    sum over t=0..H-2 of ||qdot_(t+1) - qdot_t||_2
```

```text
cost best =
    min over free trajectories of
    [cost path length_i + cost smoothness_i]
```

这三项只用于比较最终候选轨迹谁更短、谁更平滑。  
它们不是 guide 中真正优化的 `collision cost + GP cost`。

`variance waypoint` 也是后验统计量。它衡量的是：  
在每个时间点，把所有 free trajectories 的对应 waypoint 放在一起，计算它们两两之间距离的方差，再沿时间求和。  
因此它描述的是最终 free trajectories 的多样性，而不是可行性。

`t_total` 则记录的是整次推理耗时，也不进入任何代价函数。

## 5. 泛化性应该怎样理解

这个项目能说明的泛化性，主要是两类：

1. 对新起点和终点的泛化  
   训练时没见过的 start/goal，在同一环境分布下仍能采出合理轨迹。

2. 对测试时新增约束的适应  
   尤其是 Panda 推理阶段在原固定障碍物基础上再加入 extra obstacles，此时靠 guidance 将 prior 轨迹修正为新环境下更可行的轨迹。

但它不能强证明的事情也要说清楚：

- 它没有训练一个显式条件化在环境 `E` 上的生成模型
- 训练条件里只有 start/goal，没有障碍物编码
- 因此它不等于“模型本身已经学会了任意新障碍物布局”

更准确的说法是：

- prior 负责生成“像训练数据”的轨迹
- guidance 负责把轨迹适配到当前推理环境

所以这里展示的是：

- 有限的 prior 泛化
- 明确的 test-time guided adaptation

而不是纯神经网络意义上的强环境泛化。

## 6. 对 Mini2 的实际启示

对当前 Mini2 接入来说，先保持固定障碍物是合理的。

原因不是固定环境一定最好，而是当前模型定义本身就是：

```text
p_theta(X | q_s, q_g)
```

不是：

```text
p_theta(X | q_s, q_g, E)
```

如果现在直接把很多不同障碍物布局混进训练集，而不改模型输入，那么：

- 同样的 start/goal 会对应很多不同的可行轨迹分布
- 模型却看不到环境差异
- 训练会更不稳定，先验也会更模糊

因此更稳妥的顺序是：

1. 先在固定障碍物 Mini2 环境上拿到稳定 baseline  
2. 如果确实要做多环境，再考虑：
   - 少量离散环境版本
   - 或正式把环境编码进模型条件

当前 Panda 的实验更像是在证明：

- 固定环境上学到的 prior
- 配合测试时的 guidance
- 对“原环境 + 一部分新增障碍物”的情况具有适应能力

这套思路同样可以迁移到 Mini2，但前提是先把固定环境版本跑稳。

补充一点：训练时通常不需要手工修改模型输入维度。原因是训练脚本会从数据集自动读取 `state_dim`，而 `TemporalUnet` 直接用这个 `state_dim` 作为轨迹输入通道。需要区分的是，`state_dim` 是真实状态维度，随机器人和是否包含速度自动变化；`unet_input_dim` 只是 U-Net 的隐藏通道基宽，不是轨迹状态维度。因此从 Panda 切到 Mini2 时，真实输入维度会随数据集自动变化，但不需要因为机器人从 7 轴变成 6 轴而手工改 `unet_input_dim`。

## 7. 训练记录口径

这一节只记三类信息：

- 指标定义
- 当前设置值
- 为什么这样设定

### 7.1 当前 Mini2 训练设置

对应 2026-04-27 这一轮 Mini2 训练。

- 数据源：`scripts/generate_data/logs/generate_trajectories_mini2_2026-04-23_16-55-24/.../robot_id___RobotMini2`
- 数据集：`EnvSpheres3DMini2-RobotMini2`
- 总轨迹数：`9625`
- 单条轨迹形状：`(64, 12)`
- `include_velocity=True`
  原因：状态写成 `[q, qdot]`，便于模型直接学习位置和速度联合分布
- `q_dim=6`，`state_dim=12`
- 条件 `task` 维度：`12`
  原因：只用起点位置和终点位置作为条件，即 `[q_s, q_g]`
- 归一化：`LimitsNormalizer`
  原因：把每一维映到 `[-1, 1]`，让扩散训练更稳定
- 硬约束：固定第 `0` 和第 `H-1` 个 waypoint
  原因：起点终点必须严格满足

### 7.2 train / val 划分

- 划分方式：`random_split(full_dataset, [0.95, 0.05])`
- 当前结果：`train=9144`，`val=481`
- 当前切分单位：`trajectory`
- 不是：`context`

为什么这样记很重要：

- `trajectory` 级切分更适合看训练是否稳定、模型容量是否合适
- 但它不能严格代表 unseen-context 泛化

原因是同一个 context 下可能有多条轨迹，当前切分可能把同一 context 的不同轨迹同时分到 train 和 val。

因此当前验证更适合解释为：

- 容量比较
- 过拟合监控

而不是：

- 严格的新任务泛化评测

### 7.3 优化与训练过程设置

- 优化器：`Adam`
  原因：扩散模型里是常见稳定基线
- 学习率：`3e-4`
  原因：这是当前仓库用于这轮 Mini2 baseline 的直接训练配置
- 自动求导：PyTorch `autograd`
- 反向传播：`loss.backward()`
- AMP：`use_amp=False`
  原因：优先保持数值行为简单稳定
- 梯度裁剪：`clip_grad=True`
- 裁剪阈值：`max_norm=1.0`
  原因：抑制偶发大梯度，减少训练不稳定
- EMA：`use_ema=True`
- `ema_decay=0.995`
- `step_start_ema=1000`
- `update_ema_every=10`
  原因：EMA 用来平滑参数，通常比最后一步裸参数更稳
- 扩散步数：`n_diffusion_steps=25`
- 噪声调度：`variance_schedule='exponential'`
  原因：这是当前 MPD 训练脚本的实际设置
- 预测目标：`predict_epsilon=True`
  原因：直接预测噪声是 DDPM 类训练的标准形式

补充：

- 标量 `VALIDATION diffusion_loss` 用普通 `model` 算
- summary 里的采样指标和可视化优先用 `ema_model`

### 7.4 记录频率与实际训练步数

- `steps_til_summary=20000`
  原因：每 `20000` step 记录一次标量和采样结果，减少日志开销
- `steps_til_ckpt=50000`
  原因：每 `50000` step 存一次 checkpoint，方便恢复和对比
- 名义训练步数：`500000`
- 实际训练结束步数：`478800`

实际不是 `500000` 的原因：

训练代码先算：

```text
epochs = ceil(num_train_steps * batch_size / len(full_dataset))
```

当前：

- `len(full_dataset)=9625`
- `batch_size=128`
- `epochs=6650`
- `train_dataloader` 每个 epoch 有 `72` 个 batch

所以最终总步数是：

```text
6650 * 72 = 478800
```

### 7.5 `diffusion_loss` 的定义

当前 loss：

- 类型：`l2`
- 训练目标：预测噪声 `epsilon`

定义：

```text
t ~ Uniform{0, ..., T-1}
epsilon ~ N(0, I)
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
diffusion_loss = E[ || epsilon_theta(x_t, t, context) - epsilon ||^2 ]
```

这里的 `context` 当前就是：

- 起点 `q_s`
- 终点 `q_g`

为什么这样定义：

- 这是 DDPM/score-based 模型最标准、最稳定的训练形式之一
- 模型学到的是“给定起终点后，如何从噪声中恢复合理轨迹分布”

### 7.6 `VALIDATION diffusion_loss` 为什么是平均值

定义：

- 验证阶段对若干个 validation batch 分别算 `diffusion_loss`
- 最后对这些 batch loss 取平均

用途：

1. 降低单个 batch 随机性的影响
2. 更稳定地看 train / val 是否分叉
3. 便于比较不同模型配置

代码里默认还有一个上限：

- `steps_per_validation=10`

由于 `step_val` 从 `0` 开始，最多会统计前 `11` 个 validation batch。  
这样设定的原因是：

- 如果验证集很大，不必每次 summary 都跑完整个验证集
- 可以减少训练时的验证开销

但对 2026-04-27 这轮 Mini2 训练：

- `val=481`
- `batch_size=128`
- validation batch 总数只有 `4`

所以这轮的：

- `VALIDATION diffusion_loss`

实际上是：

- 对整个 validation set 的全量平均

不是只平均一部分。

### 7.7 训练期间记录的指标

#### 7.7.1 `diffusion_loss`

定义：

- 当前训练 batch 上的噪声预测 MSE

用途：

- 监控优化是否正常
- 看模型是否还在继续拟合训练分布

#### 7.7.2 `VALIDATION diffusion_loss`

定义：

- validation batch 平均后的噪声预测 MSE

用途：

- 监控过拟合
- 比较模型容量
- 看收敛情况

#### 7.7.3 `TRAINING / VALIDATION percentage free trajs`

定义：

- summary 阶段随机抽一个 subset 内的 task/context
- 用这个 start / goal 采样 `25` 条轨迹
- 统计其中 free trajectory 的比例

公式：

```text
percentage free trajs = n_free / 25
```

用途：

- 粗看模型采样后能生成多少可行轨迹

注意：

- 它不是全训练集或全验证集的总体平均

#### 7.7.4 `TRAINING / VALIDATION percentage collision intensity`

定义：

- 采样轨迹中，发生碰撞的 waypoint 数量占总 waypoint 数量的比例

公式：

```text
collision_intensity =
number of colliding waypoints / total number of waypoints
```

用途：

- 看碰撞是“偶发”还是“大面积存在”

#### 7.7.5 `TRAINING / VALIDATION success`

定义：

```text
success = 1, if at least one sampled trajectory is free
success = 0, otherwise
```

用途：

- 看当前这个被抽到的 task/context 是否至少被解出一次

#### 7.7.6 `joint trajectories DATASET`

定义：

- 真实数据轨迹的关节空间图

用途：

- 看训练数据本身长什么样

#### 7.7.7 `joint trajectories DIFFUSION`

定义：

- 模型采样轨迹的关节空间图

用途：

- 人工检查采样结果是否发散、抖动、偏离起终点

### 7.8 当前这轮结果应如何解读

当前这轮训练结果主要能支持：

1. Mini2 数据可以用于稳定训练
2. 3-stage 与 4-stage U-Net 可以直接比较容量与稳定性
3. 模型已经在学习 Mini2 固定环境下的轨迹先验

当前这轮训练结果不应直接被解释为：

1. 对 unseen-context 的严格泛化结论
2. 对新任务分布的最终 benchmark

原因只有一个最关键：

- 当前验证切分单位是 `trajectory`，不是 `context`
