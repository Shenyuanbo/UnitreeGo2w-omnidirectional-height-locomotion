# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2WRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 # 每个环境rollout24步后迭代一次ppo
    max_iterations = 20000 # 最多训练20000次迭代
    save_interval = 100 # 每100次迭代保存一次模型
    experiment_name = "unitree_go2w_rough" # 实验/日志名称
    empirical_normalization = False 

    # 定义Actor-Critic网络结构
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0, #初始动作探索噪声
        actor_hidden_dims=[512, 256, 128], # Actor网络隐藏层维度
        critic_hidden_dims=[512, 256, 128], # Critic网络隐藏层维度
        activation="elu", # Actor和Critic网络激活函数
    )

    # 定义PPO算法超参数
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, # Critic网络损失函数权重
        use_clipped_value_loss=True, # 是否使用clip的价值损失
        clip_param=0.2, # PPO clip范围
        entropy_coef=0.01, # 熵奖励，鼓励探索
        num_learning_epochs=5, # 每批 rollout 数据训练 5 轮
        num_mini_batches=4, # 数据分成 4 个 mini-batch
        learning_rate=1.0e-3,
        schedule="adaptive", # 学习率调度策略
        gamma=0.99, # 折扣因子
        lam=0.95, # GAE参数
        desired_kl=0.01, # 期望的KL散度，用于自适应学习率调度
        max_grad_norm=1.0, # 梯度裁剪，防止梯度爆炸
    )


@configclass
class UnitreeGo2WFlatPPORunnerCfg(UnitreeGo2WRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "unitree_go2w_flat"


# Task 1: Crawl_demo
@configclass
class UnitreeGo2CrawlPPORunnerCfg(UnitreeGo2WRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.experiment_name = "unitree_go2w_crawl"

# Task 2: Height_demo
@configclass
class UnitreeGo2WHeightPPORunnerCfg(UnitreeGo2WRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.experiment_name = "unitree_go2w_height"
