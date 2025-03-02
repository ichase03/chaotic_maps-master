#%% ---------------
import numpy as np
import matplotlib.pyplot as plt
import chaotic_maps as cm
from scipy.stats import norm
from collections import Counter
from tqdm import tqdm  # 导入 tqdm 库

plt_title_fontsize = 20
plt_sub_title_fontsize = 18
# 全局设置字体为新罗马
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams['font.family'] = '宋体'
plt.rcParams['font.family'] = 'SimSun'
# 全局设置字体和文字大小
plt.rcParams["font.size"] = 10  # 默认字体大小
plt.rcParams["axes.titlesize"] = 10  # 标题字体大小
plt.rcParams["axes.labelsize"] = 10  # 坐标轴标签字体大小
plt.rcParams["xtick.labelsize"] = 10  # X轴刻度标签字体大小
plt.rcParams["ytick.labelsize"] = 10  # Y轴刻度标签字体大小
plt.rcParams["legend.fontsize"] = 10  # 图例字体大小

# % 待拟合分布参数
mu1    = -0.75
sigma1 = 0.25
alpha1 = 0.2
mu2    = -0.5
sigma2 = 1.3
alpha2 = 0.2
mu3    = 0.5
sigma3 = 0.94
alpha3 = 0.2
mu4    = 0.15
sigma4 = 0.25
alpha4 = 0.1
mu5    = 0.8
sigma5 = 0.6
alpha5 = 0.1
mu6    = 1.4
sigma6 = 0.4
alpha6 = 0.2
# %代拟和分布图像
X = np.linspace(-4, 4, 1000 )
y1 = alpha1 * norm.pdf(X, mu1, sigma1)
y2 = alpha2 * norm.pdf(X, mu2, sigma2)
y3 = alpha3 * norm.pdf(X, mu3, sigma3)
y4 = alpha4 * norm.pdf(X, mu4, sigma4)
y5 = alpha5 * norm.pdf(X, mu5, sigma5)
y6 = alpha6 * norm.pdf(X, mu6, sigma6)
Y  = y1 + y2 + y3 + y4 + y5 + y6
X_NEW = list(range(1, 1001))
Y_NEW =  np.empty(1000)
Y_max = max(Y)
Y_min = min(Y)
for i in X_NEW:
    Y_NEW[i-1] = int( ( Y[i-1] - Y_min)/(Y_max - Y_min )*(2**6) )

fig, ( ax1, ax2 ) = plt.subplots(1, 2,figsize=(13/2.54, 4/2.54))
PDF1,= ax1.plot(X,Y,linewidth=1,color='r')
PDF1,= ax2.plot(X_NEW,Y_NEW,linewidth=1,color='b')
ax1.set_title('原PDF曲线')
# plt.xticks([-0.03, 0.03])
ax2.set_yticks([0,2,4,6,8,10,12,14,16] )
ax2.set_title('离散量化后的PDF曲线')
plt.subplots_adjust(wspace=0.15, hspace=0.40)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.85)
plt.savefig('../pic/PIC_draw_341.png',dpi=300)

plt.show()

random_num =  np.random.randint(0, 1000,2000 )
random_num_len = len(random_num)
sample_RAM = np.array([])
for i in tqdm(range(random_num_len), desc="Transient Phase"):
# for i in range(random_num_len):
    addr = int( random_num[i] )
    addr_value = int ( Y_NEW[ addr ] )
    new_sample = [addr] * addr_value
    sample_RAM = np.append( sample_RAM, new_sample )

count = Counter(sample_RAM)
numbers = list(count.keys())
frequencies = list(count.values())

fig, ( ax1 ) = plt.subplots(1, 1,figsize=(6/2.54, 3/2.54))
ax1.bar(numbers, frequencies)
ax1.get_yaxis().set_visible(False)
ax1.set_xticks([0,256,512,768,1024] )
plt.tight_layout()
fig.subplots_adjust(left=0.05, right=0.93, bottom=0.20, top=0.95)
plt.savefig('../pic/PIC_draw_341_2.png',dpi=300)
plt.show()
