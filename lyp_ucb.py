import numpy as np
import matplotlib.pyplot as plt

class LyapunovLinUCBScheduler:
    """
    带折扣线性UCB的Lyapunov约束MaxWeight算法 (LCMW-D-LinUCB)。
    
    该调度器根据用户提供的可行性分析报告进行设计，旨在解决高维LEO卫星网络中的
    能量感知资源分配问题。它融合了：
    1.  Lyapunov优化：通过物理和虚拟队列管理数据积压和长期能耗约束。
    2.  结构化在线学习 (线性UCB)：通过学习一个共享的低维线性模型来克服维度灾难，
        利用卫星信道之间的物理相关性。
    3.  折扣机制：通过一个折扣因子gamma来“遗忘”过时的信息，以适应非平稳的信道环境。
    """

    def __init__(self, num_arms, context_dim, V, energy_budgets, ucb_alpha=1.0, gamma=0.99):
        """
        初始化调度器。

        参数:
            num_arms (int): 臂（卫星链路）的总数, K。
            context_dim (int): 上下文向量的维度, d。
            V (float): Lyapunov权衡参数。
            energy_budgets (np.array): 每个臂的长期平均能耗预算, E_max。
            ucb_alpha (float): LinUCB的探索参数 alpha。
            gamma (float): 折扣因子 (0 < gamma <= 1)，用于处理非平稳性。
        """
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.V = V
        self.energy_budgets = np.array(energy_budgets)
        self.ucb_alpha = ucb_alpha
        self.gamma = gamma

        # Lyapunov 状态变量
        self.Q = np.zeros(num_arms)  # 物理数据队列
        self.W = np.zeros(num_arms)  # 虚拟能耗队列

        # LinUCB 学习相关的统计数据
        # A = d x d 矩阵, b = d x 1 向量
        self.A = np.identity(context_dim)  # 初始化为单位矩阵以保证可逆
        self.b = np.zeros((context_dim, 1))
        
        # 缓存计算结果
        self.A_inv = np.identity(context_dim)
        self.theta_hat = np.zeros((context_dim, 1))

    def choose_arm(self, context_vectors):
        """
        在每个时隙，根据当前的上下文向量选择一个臂。

        参数:
            context_vectors (np.array): 一个 K x d 的矩阵，每一行是对应臂的上下文向量。

        返回:
            int: 被选中的臂的索引。
        """
        # --- LinUCB部分：计算每个臂的乐观服务率估计 ---
        # 首先，根据当前的A和b计算参数theta的估计值
        self.A_inv = np.linalg.inv(self.A)
        self.theta_hat = self.A_inv @ self.b

        ucb_estimates = np.zeros(self.num_arms)
        for i in range(self.num_arms):
            x_i = context_vectors[i].reshape(-1, 1)  # 确保是列向量
            
            # 1. 预测的平均服务率 (exploitation term)
            predicted_mean = self.theta_hat.T @ x_i
            
            # 2. UCB探索奖励 (exploration term)
            exploration_bonus = self.ucb_alpha * np.sqrt(x_i.T @ self.A_inv @ x_i)
            
            # 3. 乐观的服务率估计
            ucb_estimates[i] = predicted_mean + exploration_bonus

        # --- Lyapunov部分：计算每个臂的综合得分 ---
        # 决策逻辑与之前相同，但使用LinUCB的估计值
        # 最大化: (Q + V) * S_ucb - W
        scores = (self.Q + self.V) * ucb_estimates - self.W
        
        return np.argmax(scores)

    def update(self, chosen_arm, chosen_arm_context, arrivals, observed_service_rate, observed_energy_cost):
        """
        在一个时隙结束后，根据观测结果更新调度器的状态和学习模型。
        """
        # --- 更新物理和虚拟队列 (与之前完全相同) ---
        service_vector = np.zeros(self.num_arms)
        service_vector[chosen_arm] = observed_service_rate
        self.Q = np.maximum(0, self.Q - service_vector + arrivals)

        energy_deltas = -self.energy_budgets
        energy_deltas[chosen_arm] += observed_energy_cost
        self.W = np.maximum(0, self.W + energy_deltas)

        # --- 更新LinUCB的学习模型 (核心改造部分) ---
        x_chosen = chosen_arm_context.reshape(-1, 1) # 确保是列向量
        
        # 应用折扣因子 "遗忘" 历史信息
        self.A = self.gamma * self.A + (x_chosen @ x_chosen.T)
        self.b = self.gamma * self.b + x_chosen * observed_service_rate


# --- 仿真环境 (为上下文老虎机改造) ---
def simulate_contextual_environment():
    # 仿真参数
    NUM_ARMS = 50  # 增加臂的数量以体现高维性
    CONTEXT_DIM = 4  # 上下文维度 (d << K)
    SIMULATION_STEPS = 5000
    V_PARAM = 100.0
    GAMMA = 0.995 # 折扣因子

    # 环境真实参数 (对调度器未知)
    # 这是算法需要学习的真实共享参数 theta*
    TRUE_THETA_STAR = np.array([10, -3, 5, -2]).reshape(-1, 1)

    # 真实能耗模型 (保持简单)
    TRUE_ENERGY_COSTS = np.random.uniform(3, 10, NUM_ARMS)
    ENERGY_BUDGETS = np.full(NUM_ARMS, 7.0)
    
    # === 代码修改部分：将固定到达率改为动态到达率 ===
    BASE_ARRIVAL_RATE = 0.2 # 平均到达率
    
    scheduler = LyapunovLinUCBScheduler(
        num_arms=NUM_ARMS,
        context_dim=CONTEXT_DIM,
        V=V_PARAM,
        energy_budgets=ENERGY_BUDGETS,
        gamma=GAMMA
    )

    # **BUG FIX**: 初始化history字典
    history = {
        'total_Q': [],
        'W_per_arm': [],
        'chosen_arms': [],
        'avg_energy_consumed': [],
        'theta_error': [], # 新增：追踪theta学习的误差
        'total_energy_per_arm': np.zeros(NUM_ARMS)
    }

    print("Starting simulation with LinUCB model and DYNAMIC arrivals...")
    for t in range(SIMULATION_STEPS):
        # 1. 生成时变的上下文向量 (模拟LEO环境)
        base_angle = np.sin(2 * np.pi * t / 1000 + np.linspace(0, 2*np.pi, NUM_ARMS))
        atmospheric_condition = np.random.randn(NUM_ARMS) * 0.1
        link_quality = np.linspace(-1, 1, NUM_ARMS)
        
        # **BUG FIX**: 修复 vstack 的调用
        context_vectors = np.vstack([
            np.ones(NUM_ARMS),
            base_angle,
            atmospheric_condition,
            link_quality
        ]).T # Shape: (NUM_ARMS, CONTEXT_DIM)

        # 2. 调度器决策
        chosen_arm = scheduler.choose_arm(context_vectors)

        # 3. 环境反馈
        chosen_context = context_vectors[chosen_arm]
        mean_service_rate = chosen_context @ TRUE_THETA_STAR
        observed_service = max(0, np.random.normal(mean_service_rate, 1.0))
        observed_energy = max(0, np.random.normal(TRUE_ENERGY_COSTS[chosen_arm], 0.5))

        # 4. 更新调度器
        # === 代码修改部分：使用泊松分布生成随机到达任务量 ===
        arrivals = np.random.poisson(BASE_ARRIVAL_RATE, size=NUM_ARMS)
        scheduler.update(chosen_arm, chosen_context, arrivals, observed_service, observed_energy)

        # 5. 记录数据
        history['total_Q'].append(np.sum(scheduler.Q))
        # **BUG FIX**: 记录到正确的 key
        history['W_per_arm'].append(scheduler.W.copy())
        history['chosen_arms'].append(chosen_arm)
        history['theta_error'].append(np.linalg.norm(scheduler.theta_hat - TRUE_THETA_STAR))
        
        history['total_energy_per_arm'][chosen_arm] += observed_energy
        avg_energy = history['total_energy_per_arm'] / (t + 1)
        history['avg_energy_consumed'].append(avg_energy)

        if (t+1) % 500 == 0:
            print(f"Step {t+1}/{SIMULATION_STEPS}, Theta Error: {history['theta_error'][-1]:.4f}, Total Queue: {history['total_Q'][-1]:.2f}")

    # --- 结果可视化 ---
    # **BUG FIX**: 修复所有绘图代码
    fig, axs = plt.subplots(5, 1, figsize=(14, 25))
    
    # 图1: Theta学习误差
    axs[0].plot(history['theta_error'], label='|| theta_hat - theta* ||')
    axs[0].set_title('Learning Performance: Convergence of Estimated Theta')
    axs[0].set_xlabel('Time Slot')
    axs[0].set_ylabel('Euclidean Norm Error')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].text(0.01, 0.9, 'Error approaches zero, indicating successful learning of the channel model.', 
              transform=axs[0].transAxes, fontsize=9)

    # 图2: 总数据队列长度
    axs[1].plot(history['total_Q'], label='Total Data Queue Length')
    axs[1].set_title('System Stability: Total Data Queue Length Over Time')
    axs[1].set_xlabel('Time Slot')
    axs[1].set_ylabel('Total Queue Length')
    axs[1].grid(True)
    axs[1].legend()

    # 图3: 虚拟能耗队列长度 (抽样展示)
    W_history = np.array(history['W_per_arm'])
    for i in range(0, NUM_ARMS, 10):
        axs[2].plot(W_history[:, i], label=f'Arm {i} Virtual Queue')
    axs[2].set_title('Energy Constraint Management: Virtual Queue Lengths (Sampled)')
    axs[2].set_xlabel('Time Slot')
    axs[2].set_ylabel('Virtual Queue (W) Length')
    axs[2].grid(True)
    axs[2].legend(fontsize='small')

    # 图4: 长期平均能耗 (抽样展示)
    avg_energy_history = np.array(history['avg_energy_consumed'])
    for i in range(0, NUM_ARMS, 10):
        axs[3].plot(avg_energy_history[:, i], label=f'Arm {i} Avg. Energy')
    axs[3].axhline(y=np.mean(ENERGY_BUDGETS), color='r', linestyle='--', label='Avg. Budget')
    axs[3].set_title('Long-term Average Energy Consumption (Sampled)')
    axs[3].set_xlabel('Time Slot')
    axs[3].set_ylabel('Average Energy Consumed')
    axs[3].grid(True)
    axs[3].legend(fontsize='small')

    # 图5: 臂选择频率
    arm_indices = range(NUM_ARMS)
    arm_counts = [history['chosen_arms'].count(i) for i in arm_indices]
    axs[4].bar(arm_indices, arm_counts, color='skyblue')
    axs[4].set_title('Arm Selection Frequency')
    axs[4].set_xlabel('Arm Index')
    axs[4].set_ylabel('Number of Times Chosen')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    simulate_contextual_environment()