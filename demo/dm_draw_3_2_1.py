import numpy as np
import matplotlib.pyplot as plt
import math

# from demo.draw_2_3_2_xiangtu import x_new

# 全局设置字体为新罗马
plt.rcParams["font.family"] = "Times New Roman"
# 全局设置字体和文字大小
plt.rcParams["font.size"] = 10  # 默认字体大小
plt.rcParams["axes.titlesize"] = 10  # 标题字体大小
plt.rcParams["axes.labelsize"] = 10  # 坐标轴标签字体大小
plt.rcParams["xtick.labelsize"] = 10  # X轴刻度标签字体大小
plt.rcParams["ytick.labelsize"] = 10  # Y轴刻度标签字体大小
plt.rcParams["legend.fontsize"] = 10  # 图例字体大小

# 定义忆阻器模型参数
alpha = 1.0
beta = 0.1
gamma = 0.01
h1 = 10
h2 = 0.1
Ts = 0.05
A0  = 0.01
A1  = 0.02
A2  = 0.03
fn0 = 0.01
fn1 = 0.01
fn2 = 0.01

# 模拟参数
T = 100  # 总时间
dt = 0.01  # 时间步长
t = np.arange(0, T, dt)  # 时间向量

# 生成输入电压信号（正弦波）
V0 =  A0 * np.sin(2 * np.pi * fn0 * t)
V1 =  A1 * np.sin(2 * np.pi * fn1 * t)
V2 =  A2 * np.sin(2 * np.pi * fn2 * t)

# 初始化忆阻器状态
x0 = np.zeros_like(t)
i0 = np.zeros_like(t)
x1 = np.zeros_like(t)
i1 = np.zeros_like(t)
x2 = np.zeros_like(t)
i2 = np.zeros_like(t)

# 模拟忆阻器响应
for n in range(1, len(t)):
    # # 更新忆阻器状态
    # dx = alpha * V[n - 1] - beta * x[n - 1] - gamma * x[n - 1]**3
    # x[n] = x[n - 1] + dx * dt
    # # 计算忆阻器电流
    # i[n] = x[n] * V[n]

    # i0[n] = ( h1 * (x0[n-1] ** 2) + h2 * (x0[n-1] ** 4) ) * V0[n-1]
    i0[n] = math.sin(x0[n-1]) * V0[n-1]
    x0[n] = x0[n-1] + Ts * V0[n-1]
for n in range(1, len(t)):
    # # 更新忆阻器状态
    # dx = alpha * V[n - 1] - beta * x[n - 1] - gamma * x[n - 1]**3
    # x[n] = x[n - 1] + dx * dt
    # # 计算忆阻器电流
    # i[n] = x[n] * V[n]

    # i1[n] = ( h1 * x1[n-1] ** 2 + h2 * x1[n-1] ** 4 ) * V1[n-1]
    i1[n] = math.sin(x1[n-1]) * V1[n-1]
    x1[n] = x1[n-1] + Ts * V1[n-1]
for n in range(1, len(t)):
    # # 更新忆阻器状态
    # dx = alpha * V[n - 1] - beta * x[n - 1] - gamma * x[n - 1]**3
    # x[n] = x[n - 1] + dx * dt
    # # 计算忆阻器电流
    # i[n] = x[n] * V[n]

    # i2[n] = (h1 * x2[n - 1]  ** 2 + h2 * x2[n - 1]  ** 4 ) * V2[n - 1]
    i2[n] = math.sin(x2[n-1]) * V2[n-1]
    x2[n] = x2[n - 1] + Ts * V2[n - 1]

# 绘制磁滞曲线
plt.figure(figsize=(7/2.54, 6/2.54))
plt.plot(V0, i0,'r',label='0.01Hz')
plt.plot(V1, i1,'b',label='0.02Hz')
plt.plot(V2, i2,'g',label='0.03Hz')
plt.xticks([-0.03, 0.03])
plt.yticks([-0.03,0.03])
plt.grid(False)
# plt.tick_params(axis='x', bottom=False,top=False )
# plt.tick_params(axis='both', length=0)
plt.xlabel('$u_n(V)$',labelpad=-10)
plt.ylabel('$i_n(A)$',labelpad=-15)

# plt.title('Discrete Memristor Hysteresis Curve')
plt.tight_layout()
plt.legend()
# plt.grid(True)
plt.savefig('../pic/PIC_321_DM.png',dpi=300)
plt.show()