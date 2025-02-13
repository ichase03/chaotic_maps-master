# # import numpy as np
# # from scipy.integrate import solve_ivp
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# #
# # # Rössler系统的微分方程
# # def rossler_system(t, state, a, b, c):
# #     x, y, z = state
# #     dxdt = -y - z
# #     dydt = x + a * y
# #     dzdt = b + z * (x - c)
# #     return [dxdt, dydt, dzdt]
# #
# # # 参数
# # a, b, c = 0.2, 0.2, 5.7
# #
# # # 初始条件
# # initial_state = [0.0, 2.0, 0.0]
# #
# # # 时间范围
# # t_span = (0, 500)
# # t_eval = np.linspace(t_span[0], t_span[1], 20000)
# #
# # # 求解微分方程
# # sol = solve_ivp(rossler_system, t_span, initial_state, args=(a, b, c), t_eval=t_eval)
# #
# # # 提取解
# # x, y, z = sol.y
# #
# # # 绘制三维相图
# # fig = plt.figure(figsize=(4, 4))
# # ax = fig.add_subplot(111, projection='3d')
# #
# # # 去掉背景
# # ax.grid(False)  # 去掉网格
# # ax.xaxis.pane.fill = False  # 去掉x轴背景面
# # ax.yaxis.pane.fill = False  # 去掉y轴背景面
# # ax.zaxis.pane.fill = False  # 去掉z轴背景面
# #
# # # 去掉坐标轴背景线
# # # ax.xaxis.pane.set_edgecolor('w')
# # # ax.yaxis.pane.set_edgecolor('w')
# # # ax.zaxis.pane.set_edgecolor('w')
# #
# # # 绘制轨迹
# # ax.plot(x, y, z, lw=0.5, color='b')
# #
# # # 设置观测角度
# # ax.view_init(elev=20, azim=210)  # 仰角20度，方位角30度
# #
# # # 设置标签
# # ax.set_xlabel('X')
# # ax.set_ylabel('Y')
# # ax.set_zlabel('Z')
# # # ax.set_title('Rössler Attractor')
# #
# # plt.savefig("PIC_2_3_2_相图.png", dpi=100)
# # plt.show()
# #
# #
import numpy as np
import matplotlib.pyplot as plt

# Hénon映射的参数
a = 1.4
b = 0.3

# 初始条件
x0, y0 = 0, 0

# 迭代次数
num_iterations = 10000

# 存储轨迹
x_values = []
y_values = []

# 初始化
x, y = x0, y0

# 迭代Hénon映射
for _ in range(num_iterations):
    x_new = 1 - a * x**2 + y
    y_new = b * x
    x, y = x_new, y_new
    x_values.append(x)
    y_values.append(y)

# 绘制相图
plt.figure(figsize=(4, 3))
plt.scatter(x_values, y_values, s=0.1, color='blue', alpha=0.5)
# plt.title("Hénon Map")
plt.xlabel("x")
plt.ylabel("y")
plt.subplots_adjust(left=0.17 ,right=0.95, bottom=0.15, top=0.95)
plt.savefig("PIC_2_3_2_相图1.png", dpi=100)
plt.show()
# #
# #
# #
# # # 创建画布和子图
# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# #
# # # 绘制Hénon映射
# # ax1.scatter(x_values, y_values, s=0.1, color='blue', alpha=0.5)
# # ax1.set_title("Hénon Map")
# # ax1.set_xlabel("x")
# # ax1.set_ylabel("y")
# # ax1.grid(False)  # 去掉网格
# #
# # # 绘制Rössler吸引子
# # ax2 = fig.add_subplot(122, projection='3d')
# # ax2.plot(x_rossler, y_rossler, z_rossler, lw=0.5, color='b')
# # ax2.set_title("Rössler Attractor")
# # ax2.set_xlabel('X')
# # ax2.set_ylabel('Y')
# # ax2.set_zlabel('Z')
# # ax2.grid(False)  # 去掉网格
# #
# # # 调整布局
# # plt.tight_layout()
# # plt.show()
#
# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # ========================================
# # Hénon映射
# # ========================================
#
# # Hénon映射的参数
# a_henon = 1.4
# b_henon = 0.3
#
# # 初始条件
# x0, y0 = 0, 0
#
# # 迭代次数
# num_iterations = 10000
#
# # 存储轨迹
# x_values = []
# y_values = []
#
# # 初始化
# x, y = x0, y0
#
# # 迭代Hénon映射
# for _ in range(num_iterations):
#     x_new = 1 - a_henon * x**2 + y
#     y_new = b_henon * x
#     x, y = x_new, y_new
#     x_values.append(x)
#     y_values.append(y)
#
# # ========================================
# # Rössler吸引子
# # ========================================
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
# a_rossler, b_rossler, c_rossler = 0.2, 0.2, 5.7
#
# # 初始条件
# initial_state = [0.0, 2.0, 0.0]
#
# # 时间范围
# t_span = (0, 500)
# t_eval = np.linspace(t_span[0], t_span[1], 10000)
#
# # 求解微分方程
# sol = solve_ivp(rossler_system, t_span, initial_state, args=(a_rossler, b_rossler, c_rossler), t_eval=t_eval)
#
# # 提取解
# x_rossler, y_rossler, z_rossler = sol.y
#
# # ========================================
# # 绘制图形
# # ========================================
#
# # 创建画布和子图
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
#
# # 绘制Hénon映射
# ax1.scatter(x_values, y_values, s=0.1, color='blue', alpha=0.5)
# # ax1.set_title("Hénon Map")
# ax1.set_xlabel("x")
# ax1.set_ylabel("y")
# ax1.grid(False)  # 去掉网格
#
# # 绘制Rössler吸引子
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot(x_rossler, y_rossler, z_rossler, lw=0.5, color='b')
# # ax2.set_title("Rössler Attractor")
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')
# ax2.grid(False)  # 去掉网格
# ax2.xaxis.pane.fill = False  # 去掉x轴背景面
# ax2.yaxis.pane.fill = False  # 去掉y轴背景面
# ax2.zaxis.pane.fill = False  # 去掉z轴背景面
# ax2.view_init(elev=20, azim=210)  # 仰角20度，方位角30度
#
# # 调整布局
# plt.tight_layout()
# plt.show()
#
