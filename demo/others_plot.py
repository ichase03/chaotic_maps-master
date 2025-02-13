# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.stats import norm
# #
# #
# # # 目标分布：标准正态分布
# # def target_distribution(x):
# #     return norm.pdf(x)
# #
# #
# # # 提议分布：均匀分布
# # def proposal_distribution(x):
# #     return np.ones_like(x) / (4 * np.sqrt(2 * np.pi))  # 均匀分布的PDF
# #
# #
# # # 常数 M
# # M = 4  # 使得 M * g(x) >= f(x) 对所有 x 成立
# #
# #
# # # 生成随机点
# # def acceptance_rejection_demo(size):
# #     accepted_x = []
# #     rejected_x = []
# #     accepted_y = []
# #     rejected_y = []
# #
# #     while len(accepted_x) < size:
# #         Y = np.random.uniform(-4, 4)  # 从均匀分布生成 Y
# #         U = np.random.uniform(0, 1)  # 生成均匀分布的 U
# #         proposal_y = U * M * proposal_distribution(Y)
# #         target_y = target_distribution(Y)
# #
# #         if proposal_y <= target_y:
# #             accepted_x.append(Y)
# #             accepted_y.append(proposal_y)
# #         else:
# #             rejected_x.append(Y)
# #             rejected_y.append(proposal_y)
# #
# #     return np.array(accepted_x), np.array(rejected_x), np.array(accepted_y), np.array(rejected_y)
# #
# #
# # # 参数设置
# # size = 1000  # 接受点的数量
# #
# # # 生成随机点
# # accepted_x, rejected_x, accepted_y, rejected_y = acceptance_rejection_demo(size)
# #
# # # 绘制示意图
# # x = np.linspace(-4, 4, 1000)
# # plt.figure(figsize=(10, 6))
# # plt.plot(x, target_distribution(x), 'b',lw=6, label='Target Distribution $f(x)$')  # 目标分布
# # plt.plot(x, M * proposal_distribution(x), 'r',lw=6, label='Proposal Distribution $M \cdot g(x)$')  # 提议分布
# # plt.scatter(accepted_x, accepted_y, color='b', alpha=0.5, label='Accepted Points')  # 接受的点
# # plt.scatter(rejected_x, rejected_y, color='pink', alpha=0.5, label='Rejected Points')  # 拒绝的点
# # plt.xlabel('x')
# # plt.ylabel('Density')
# # plt.title('Acceptance-Rejection Method Demonstration')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
# #
# #
# #
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # # Logistic映射函数
# # def logistic_map(x, r):
# #     return r * x * (1 - x)
# #
# # # 分岔图
# # def bifurcation_diagram(r_values, iterations, last):
# #     x = 1e-5 * np.ones(len(r_values))  # 初始值
# #     for i in range(iterations):
# #         x = logistic_map(x, r_values)
# #         if i >= (iterations - last):  # 只保留最后几次迭代
# #             plt.plot(r_values, x, ',k', alpha=0.25)
# #     plt.xlabel('r')
# #     plt.ylabel('x')
# #     plt.title('Logistic Map Bifurcation Diagram')
# #     plt.show()
# #
# # # 参数设置
# # r_values = np.linspace(2.5, 4.0, 10000)  # r 的范围
# # iterations = 1000  # 迭代次数
# # last = 100  # 保留最后几次迭代
# #
# # # 绘制分岔图
# # bifurcation_diagram(r_values, iterations, last)
# #
# # def logistic_chaos(r, x0, n):
# #     x = [x0]
# #     for _ in range(n-1):
# #         x.append(logistic_map(x[-1], r))
# #     return x
# #
# # # 参数设置
# # r = 3.9  # 混沌参数
# # x0 = 0.5  # 初始值
# # n = 100  # 迭代次数
# #
# # # 生成混沌序列
# # chaos_sequence = logistic_chaos(r, x0, n)
# #
# # # 绘制混沌行为
# # plt.plot(chaos_sequence, 'b-')
# # plt.xlabel('Iteration')
# # plt.ylabel('x')
# # plt.title(f'Logistic Map Chaos (r = {r})')
# # plt.show()
# #
#
# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # Rössler系统的微分方程
# def rossler_system(t, state, a, b, c):
#     x, y, z = state
#     dxdt = -y - z
#     dydt = x + a * y
#     dzdt = b + z * (x - c)
#     return [dxdt, dydt, dzdt]
#
# # 参数
# a, b, c = 0.2, 0.2, 5.7
#
# # 初始条件
# initial_state = [0.0, 2.0, 0.0]
#
# # 时间范围
# t_span = (0, 500)
# t_eval = np.linspace(t_span[0], t_span[1], 20000)
#
# # 求解微分方程
# sol = solve_ivp(rossler_system, t_span, initial_state, args=(a, b, c), t_eval=t_eval)
#
# # 提取解
# x, y, z = sol.y
#
# # 绘制三维相图
# fig = plt.figure(figsize=(4, 4))
# ax = fig.add_subplot(111, projection='3d')
#
# # 去掉背景
# ax.grid(False)  # 去掉网格
# ax.xaxis.pane.fill = False  # 去掉x轴背景面
# ax.yaxis.pane.fill = False  # 去掉y轴背景面
# ax.zaxis.pane.fill = False  # 去掉z轴背景面
#
# # 去掉坐标轴背景线
# # ax.xaxis.pane.set_edgecolor('w')
# # ax.yaxis.pane.set_edgecolor('w')
# # ax.zaxis.pane.set_edgecolor('w')
#
# # 绘制轨迹
# ax.plot(x, y, z, lw=0.5, color='b')
#
# # 设置观测角度
# ax.view_init(elev=20, azim=210)  # 仰角20度，方位角30度
#
# # 设置标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# # ax.set_title('Rössler Attractor')
#
# plt.savefig("PIC_2_3_41_LOGISTIC.png", dpi=100)
# plt.show()
#


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 定义目标函数
def target_function(x):
    return np.sin(2 * np.pi * x) + 0.5 * np.random.normal(size=len(x))

# 生成数据
np.random.seed(42)
x = np.random.uniform(0, 1, 500)
y = target_function(x)

# 将数据组合成二维数据
data = np.column_stack((x, y))

# 训练GMM
n_components = 5  # 高斯分布的数量
gmm = GaussianMixture(n_components=n_components, covariance_type='full')
gmm.fit(data)

# 生成新的x值
x_new = np.linspace(0, 1, 1000)

# 预测y值
y_new = []
for xi in x_new:
    # 计算条件概率 p(y | x)
    conditional_means = []
    conditional_weights = []
    for k in range(n_components):
        mu_k = gmm.means_[k]
        sigma_k = gmm.covariances_[k]
        weight_k = gmm.weights_[k]

        # 条件分布的均值和方差
        mu_y_given_x = mu_k[1] + sigma_k[1, 0] / sigma_k[0, 0] * (xi - mu_k[0])
        var_y_given_x = sigma_k[1, 1] - sigma_k[1, 0] / sigma_k[0, 0] * sigma_k[0, 1]

        conditional_means.append(mu_y_given_x)
        conditional_weights.append(weight_k * np.exp(-0.5 * (xi - mu_k[0])**2 / sigma_k[0, 0]))

    # 加权平均
    y_new.append(np.sum(np.array(conditional_weights) * np.array(conditional_means)) / np.sum(conditional_weights))

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=5, color='blue', label='Data')
plt.plot(x_new, y_new, color='red', label='GMM Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gaussian Mixture Model Fit')
plt.legend()
plt.show()