

#%% ---------------
import numpy as np
import matplotlib.pyplot as plt
import chaotic_maps as cm
from scipy.stats import norm

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

model_alpha01 = 0.0625
model_alpha02 = 0.0625
model_alpha03 = 0.0625
model_alpha04 = 0.0625
model_alpha05 = 0.0625
model_alpha06 = 0.0625
model_alpha07 = 0.0625
model_alpha08 = 0.0625
model_alpha09 = 0.0625
model_alpha10 = 0.0625
model_alpha11 = 0.0625
model_alpha12 = 0.0625
model_alpha13 = 0.0625
model_alpha14 = 0.0625
model_alpha15 = 0.0625
model_alpha16 = 0.0625

model_mu01 = 1.623295064204183
model_mu02 = 0.8093028059961334
model_mu03 = 0.8093028059961334
model_mu04 = -0.843352766629867
model_mu05 = 0.5384131020399228
model_mu06 = 0.1656817738817678
model_mu07 = 0.6528410350079011
model_mu08 = 1.2004630299413255
model_mu09 = -0.794724249291650
model_mu10 = 1.3516433476534078
model_mu11 = -0.63100011374045
model_mu12 = -0.554454525665790
model_mu13 = -0.554454525665790
model_mu14 = -0.843352766629867
model_mu15 = 0.5384131020399228
model_mu16 = 0.1656817738817678

model_sigma01 = 0.31574309055054134
model_sigma02 = 0.794257706824986
model_sigma03 = 0.794257706824986
model_sigma04 = 0.21776743449577074
model_sigma05 = 0.9239323589621729
model_sigma06 = 0.3090212165971693
model_sigma07 = 0.8877232263094028
model_sigma08 = 0.2993481823481516
model_sigma09 = 1.0248570353356257
model_sigma10 = 0.35945943088246185
model_sigma11 = 0.18578589458595005
model_sigma12 = 1.340092675120131
model_sigma13 = 1.340092675120131
model_sigma14 = 0.21776743449577074
model_sigma15 = 0.9239323589621729
model_sigma16 = 0.3090212165971693

y1_new  = model_alpha01*norm.pdf(X, model_mu01, model_sigma01 )
y2_new  = model_alpha02*norm.pdf(X, model_mu02, model_sigma02 )
y3_new  = model_alpha03*norm.pdf(X, model_mu03, model_sigma03 )
y4_new  = model_alpha04*norm.pdf(X, model_mu04, model_sigma04 )
y5_new  = model_alpha05*norm.pdf(X, model_mu05, model_sigma05 )
y6_new  = model_alpha06*norm.pdf(X, model_mu06, model_sigma06 )
y7_new  = model_alpha07*norm.pdf(X, model_mu07, model_sigma07 )
y8_new  = model_alpha08*norm.pdf(X, model_mu08, model_sigma08 )
y9_new  = model_alpha09*norm.pdf(X, model_mu09, model_sigma09 )
y10_new = model_alpha10*norm.pdf(X, model_mu10, model_sigma10 )
y11_new = model_alpha11*norm.pdf(X, model_mu11, model_sigma11 )
y12_new = model_alpha12*norm.pdf(X, model_mu12, model_sigma12 )
y13_new = model_alpha13*norm.pdf(X, model_mu13, model_sigma13 )
y14_new = model_alpha14*norm.pdf(X, model_mu14, model_sigma14 )
y15_new = model_alpha15*norm.pdf(X, model_mu15, model_sigma15 )
y16_new = model_alpha16*norm.pdf(X, model_mu16, model_sigma16 )
Y_new   = y1_new + y2_new + y3_new + y4_new + y5_new + y6_new + y7_new + y8_new + y9_new + y10_new + y11_new + y12_new + y13_new + y14_new + y15_new + y16_new

norm_sample_01 = np.random.normal(loc=model_mu01,  scale = model_sigma01, size=1000)
norm_sample_02 = np.random.normal(loc=model_mu02,  scale = model_sigma02, size=1000)
norm_sample_03 = np.random.normal(loc=model_mu03,  scale = model_sigma03, size=1000)
norm_sample_04 = np.random.normal(loc=model_mu04,  scale = model_sigma04, size=1000)
norm_sample_05 = np.random.normal(loc=model_mu05,  scale = model_sigma05, size=1000)
norm_sample_06 = np.random.normal(loc=model_mu06,  scale = model_sigma06, size=1000)
norm_sample_07 = np.random.normal(loc=model_mu07,  scale = model_sigma07, size=1000)
norm_sample_08 = np.random.normal(loc=model_mu08,  scale = model_sigma08, size=1000)
norm_sample_09 = np.random.normal(loc=model_mu09,  scale = model_sigma09, size=1000)
norm_sample_10 = np.random.normal(loc=model_mu10,  scale = model_sigma10, size=1000)
norm_sample_11 = np.random.normal(loc=model_mu11,  scale = model_sigma11, size=1000)
norm_sample_12 = np.random.normal(loc=model_mu12,  scale = model_sigma12, size=1000)
norm_sample_13 = np.random.normal(loc=model_mu13,  scale = model_sigma13, size=1000)
norm_sample_14 = np.random.normal(loc=model_mu14,  scale = model_sigma14, size=1000)
norm_sample_15 = np.random.normal(loc=model_mu15,  scale = model_sigma15, size=1000)
norm_sample_16 = np.random.normal(loc=model_mu16,  scale = model_sigma16, size=1000)
norm_sample =  np.concatenate(( norm_sample_01 ,
                                norm_sample_02 ,
                                norm_sample_03 ,
                                norm_sample_04 ,
                                norm_sample_05 ,
                                norm_sample_06 ,
                                norm_sample_07 ,
                                norm_sample_08 ,
                                norm_sample_09 ,
                                norm_sample_10 ,
                                norm_sample_11 ,
                                norm_sample_12 ,
                                norm_sample_13 ,
                                norm_sample_14 ,
                                norm_sample_15 ,
                                norm_sample_16 ,
                               ))

norm_sample_max = np.max(norm_sample)
norm_sample_min = np.min(norm_sample)

norm_sample_norm = ( norm_sample - norm_sample_min )/(norm_sample_max - norm_sample_min)
# 绘制直方图
plt.hist(norm_sample_norm, bins=200, density=True, alpha=0.7, color='b')

# 添加标题和标签
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Density')

# 显示图形
plt.show()
with open('../txt/norm_sample_normalized.txt', 'w') as file:
    for value in norm_sample_norm[:16384]:
        file.write(str(value) + '\n')

fig, ( ax1 ) = plt.subplots(1, 1,figsize=(13/2.54, 6/2.54))
PDF1,= ax1.plot(X,Y,linewidth=5,color='r', label = 'ORI_PDF')
PDF2,= ax1.plot(X,Y_new,linewidth=2,color='b')
h1  ,= ax1.plot(X,y1_new,linewidth=1,color='b')
h2  ,= ax1.plot(X,y2_new,linewidth=1,color='b')
h3  ,= ax1.plot(X,y3_new,linewidth=1,color='b')
h4  ,= ax1.plot(X,y4_new,linewidth=1,color='b')
h5  ,= ax1.plot(X,y5_new,linewidth=1,color='b')
h6  ,= ax1.plot(X,y6_new,linewidth=1,color='b')
h7  ,= ax1.plot(X,y7_new,linewidth=1,color='b')
h8  ,= ax1.plot(X,y8_new,linewidth=1,color='b')
h9  ,= ax1.plot(X,y9_new,linewidth=1,color='b')
h10 ,= ax1.plot(X,y10_new,linewidth=1,color='b')
h11 ,= ax1.plot(X,y11_new,linewidth=1,color='b')
h12 ,= ax1.plot(X,y12_new,linewidth=1,color='b')
h13 ,= ax1.plot(X,y13_new,linewidth=1,color='b')
h14 ,= ax1.plot(X,y14_new,linewidth=1,color='b')
h15 ,= ax1.plot(X,y15_new,linewidth=1,color='b')
h16 ,= ax1.plot(X,y16_new,linewidth=1,color='b')
ax1.legend( handles=[PDF1,PDF2,h1],labels = [ '待拟合分布', 'GMM拟合结果', '子高斯分布' ], loc = 'upper left' )


plt.show()