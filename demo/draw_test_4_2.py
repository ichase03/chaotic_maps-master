# import numpy as np
# import matplotlib.pyplot as plt
# import chaotic_maps as cm
#
# plt_title_fontsize = 20
# plt_sub_title_fontsize = 18
#
# #%%
# # 模型基础参数设置
# ux1 = 0.5
# ux2 = 0.5
# ux3 = 0.1
# ux4 = 0.1
# # xstate0 = [ux1,ux2,ux3,ux4]
# #  模型已知的参数
# ua    = 1.2
# ub    = 0.1
# uc    = 0.5
# ud    = 1.72
# ue    = np.pi / 6
# uk0   = 0.1
# uk1   = -10
# uk2   = 0.5
# utao  = 1
# # parameters = [ua,ub,uc,ud,ue,uk0,uk1,utao0,utao1]
# parameters_init = np.array(( ux1,ux2,ux3,ux4,ua,ub,uc,ud,ue,uk0,uk1,uk2,utao ))
#
# # 相关控制参数
# # param_var_range = np.linspace(-1.3,1.2,300)
# param_var_range = np.linspace(0.5,1.3,300)
# chaotic_dim_num = 4
# n_iterations = 700
# param_seqnum = 4
# y_points = 500
# dropped_steps = 2000
#
# #%%
# # 分岔点图绘制
# # 计算关于参数parameters[parameter_seqnum]在param_var_range范围变动时的分岔点集
# x_list, y_list = cm.nD_bifurcation_data(
#     f = cm.mod_4d_TBMHM ,
#     init = parameters_init ,
#     parameter_range = param_var_range ,
#     parameter_seqnum = param_seqnum ,
#     y_points = y_points,
#     dropped_steps = dropped_steps
# )
#
# # 将计算结果展开为序列
# para_list = y_list[:,:,param_seqnum].flatten()
# y0_list = y_list[:,:,0].flatten()
# y1_list = y_list[:,:,1].flatten()
# y2_list = y_list[:,:,2].flatten()
# y3_list = y_list[:,:,3].flatten()
#
# # 绘制分岔图
# fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
# ax1.plot( para_list,y0_list, "r,")
# ax1.set_title("Bifurcation Diagram:x",fontsize=plt_sub_title_fontsize)
# ax1.set_xlim(param_var_range[0], param_var_range[-1] )
#
# ax2.plot( para_list,y1_list, "b,")
# ax2.set_title("Bifurcation Diagram:y",fontsize=plt_sub_title_fontsize)
# ax2.set_xlim(param_var_range[0], param_var_range[-1])
#
# ax3.plot( para_list,y2_list, "g,")
# ax3.set_title("Bifurcation Diagram:z",fontsize=plt_sub_title_fontsize)
# ax3.set_xlim(param_var_range[0], param_var_range[-1])
#
# ax4.plot( para_list,y3_list, "y,")
# ax4.set_title("Bifurcation Diagram:u",fontsize=plt_sub_title_fontsize)
# ax4.set_xlim(param_var_range[0], param_var_range[-1])
# # plt.xlabel("xlabel")
# # plt.ylabel("ylabel")
# plt.subplots_adjust(wspace=0.2, hspace=0.3)
# plt.suptitle("My Plot Title",fontsize=plt_title_fontsize)
# fig.subplots_adjust(left=0.05 ,right=0.95, bottom=0.07, top=0.88)
#
# plt.show()
#
#
# lya_list = cm.nD_chaotic_Lya_calc(
#     chaotic_map      = cm.mod_4d_TBMHM          ,
#     jacobian         = cm.mod_4d_TBMHM_Jac      ,
#     init             = parameters_init          ,
#     n_iterations     = n_iterations             ,
#     chaotic_dim_num  = chaotic_dim_num          ,
#     parameter_range  = param_var_range          ,
#     parameter_seqnum = param_seqnum
# )
#
# lya_list0 = lya_list[:,0]
# lya_list1 = lya_list[:,1]
# lya_list2 = lya_list[:,2]
# lya_list3 = lya_list[:,3]
#
# fig, ( ax1 ) = plt.subplots(1, 1, figsize=(9, 6))
#
#
# l0, = ax1.plot(param_var_range, lya_list0, "r-", linewidth=1, label = 'LE1')
# ax1.set_xlim(param_var_range[0], param_var_range[-1])
# l1, = ax1.plot(param_var_range, lya_list1, "b-", label = 'LE1')
# l2, = ax1.plot(param_var_range, lya_list2, "g-", label = 'LE1')
# l3, = ax1.plot(param_var_range, lya_list3, "y-", label = 'LE1')
#
# ax1.legend( handles=[l0,l1,l2,l3],labels = [ 'LE0', 'LE1', 'LE2', 'LE3' ], loc = 'best', fontsize = plt_sub_title_fontsize )
# plt.xlabel("param",fontsize = plt_sub_title_fontsize)
# plt.ylabel("Lya",fontsize = plt_sub_title_fontsize)
# plt.subplots_adjust(wspace=0.2, hspace=0.3)
# plt.suptitle("Lya diagram",fontsize=plt_title_fontsize)
# fig.subplots_adjust(left=0.10 ,right=0.96, bottom=0.11, top=0.91)
#
# plt.show()
#
# #%%
#
# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
#
# mean, std = 1, 0.3
# sample_size = 2000
#
# # norm distribution
# samples = stats.norm.rvs(mean, std, size=sample_size)
#
# res = stats.relfreq(samples, numbins=20)
# pdf_value = res.frequency
# cdf_value = np.cumsum(res.frequency)
#
# x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
#
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))
# ax1.set_xlabel('Frequency (MHz)')
# ax1.set_ylabel('Probability')
# ax1.hist(samples, 15, density=True)
#
# ax1.set_xlim([x.min(), x.max()])
# ax1.grid(True, linestyle='--', alpha=0.4)
# ax2.set_xlabel('Frequency (MHz)')
# ax2.set_ylabel('Cumulative Probability')
# ax2.plot(x, cdf_value)
# ax2.grid(True, linestyle='--', alpha=0.4)
#
# plt.show()
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
#
# ax1.set_xlabel('Frequency (MHz)',fontsize = 18)
# ax1.set_ylabel('Cumulative Probability',fontsize = 20)
# ax1.plot(x, cdf_value)
# ax1.grid(True, linestyle='--', alpha=0.4)
#
# ax2.set_xlabel('U',fontsize = 20)
# ax2.set_ylabel('x_i',fontsize = 20)
# ax2.plot(cdf_value,x)
# ax2.grid(True, linestyle='--', alpha=0.4)
# fig.subplots_adjust(left=0.10 ,right=0.96, bottom=0.15, top=0.91)
#
# plt.show()

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Rössler系统的微分方程
def rossler_system(t, state, a, b, c):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

# 参数
a, b, c = 0.2, 0.2, 5.7

# 初始条件
initial_state = [0.0, 2.0, 0.0]

# 时间范围
t_span = (0, 1000)
t_eval = np.linspace(t_span[0], t_span[1], 100000)

# 求解微分方程
sol = solve_ivp(rossler_system, t_span, initial_state, args=(a, b, c), t_eval=t_eval)

# 提取解
x, y, z = sol.y

# 绘制三维相图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Rössler Attractor')
plt.show()