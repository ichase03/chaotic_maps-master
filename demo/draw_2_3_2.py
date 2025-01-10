# draw 2-3-2绘图
#%% ---------------
import numpy as np
import matplotlib.pyplot as plt
import chaotic_maps as cm
plt_title_fontsize = 20
plt_sub_title_fontsize = 18


#%%
# 蛛网图绘制
fig, axes = plt.subplots(2, 2)
cm.cobweb_plot(cm.sin_map(.7), .1, 20, domain=np.linspace(0, 1), ax=axes[0][0], ylabel=r'$x_{n+1}$')
cm.cobweb_plot(cm.sin_map(.8), .1, 20, domain=np.linspace(0, 1), ax=axes[0][1])
cm.cobweb_plot(cm.sin_map(.85), .1, 20, domain=np.linspace(0, 1), ax=axes[1][0], xlabel=r'$x_n$', ylabel=r'$x_{n+1}$')
cm.cobweb_plot(cm.sin_map(.9), .1, 20, domain=np.linspace(0, 1), ax=axes[1][1], xlabel=r'$x_n$')
axes[0][0].set_title(r'$\mu = 0.7$')
axes[0][1].set_title(r'$\mu = 0.8$')
axes[1][0].set_title(r'$\mu = 0.85$')
axes[1][1].set_title(r'$\mu = 0.9$')
plt.tight_layout()
plt.show()
# # plt.savefig("sin_cobwebs.png", dpi=300)

#%%
# 分岔图➕LE
parameters_init = np.array(( .2,.8 ))

# 相关控制参数
param_var_range = np.linspace(0.6,1,300)
param_seqnum = 1
y_points = 200
dropped_steps = 2000
chaotic_dim_num = 1
n_iterations = 70

# 计算关于参数parameters[parameter_seqnum]在param_var_range范围变动时的分岔点集
x_list, y_list = cm.nD_bifurcation_data(
    f = cm.sin_map_new ,
    init = parameters_init ,
    parameter_range = param_var_range ,
    parameter_seqnum = param_seqnum ,
    y_points = y_points,
    dropped_steps = dropped_steps
)

# 将分岔点集计算结果展开为序列
para_list = y_list[:,:,param_seqnum].flatten()
y0_list = y_list[:,:,0].flatten()


lya_list = cm.nD_chaotic_Lya_calc(
    chaotic_map      = cm.sin_map_new       ,
    jacobian         = cm.sin_map_deriv_new   ,
    init             = parameters_init          ,
    n_iterations     = n_iterations             ,
    chaotic_dim_num  = chaotic_dim_num          ,
    parameter_range  = param_var_range          ,
    parameter_seqnum = param_seqnum
)

lya_list0 = lya_list[:,0]

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot( para_list,y0_list, "r,")
ax1.set_title("Bifurcation Diagram:x",fontsize=plt_sub_title_fontsize)
ax1.set_xlim(param_var_range[0], param_var_range[-1] )

ax2.plot(param_var_range, lya_list0, "r-", linewidth=1, label = 'LE1')
ax2.plot(param_var_range, param_var_range*0, "k--", linewidth=0.5 )
ax2.set_xlim(param_var_range[0], param_var_range[-1])

# plt.xlabel("xlabel")
# plt.ylabel("ylabel")
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.suptitle("My Plot Title",fontsize=plt_title_fontsize)
fig.subplots_adjust(left=0.05 ,right=0.95, bottom=0.07, top=0.88)

plt.show()



# #%%
# #
# # 模型基础参数设置
# ux1 = 0.1
# ux2 = 0.1
# ux3 = 0.1
# ux4 = 0.1
# # xstate0 = [ux1,ux2,ux3,ux4]
# #  模型已知的参数
# ua    = 1.7
# ub    = 0.5
# uc    = 0.5
# ud    = 0.5
# ue    = 0.5
# uk0   = 0.5
# uk1   = 0.5
# utao0 = 0.5
# utao1 = 0.5
# # parameters = [ua,ub,uc,ud,ue,uk0,uk1,utao0,utao1]
# parameters_init = np.array(( ux1,ux2,ux3,ux4,ua,ub,uc,ud,ue,uk0,uk1,utao0,utao1 ))
#
# # 控制参数
# calc_num = 50000
#
# model_param_dim = len(parameters_init)
# state_list = np.zeros([calc_num,model_param_dim])
# cur_state = parameters_init
# for i in range(calc_num):
#     nxt_xtate = cm.nD_chaotic_func( cur_state )
#     state_list[i] = nxt_xtate
#     cur_state = nxt_xtate
#
# xtate0_list = state_list[:,0].flatten()
# xtate1_list = state_list[:,1].flatten()
# xtate2_list = state_list[:,2].flatten()
# xtate3_list = state_list[:,3].flatten()
#
# fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
# ax1.plot( xtate0_list,xtate1_list, "r,")
# ax1.set_title("Bifurcation Diagram:x",fontsize=plt_sub_title_fontsize)
# # ax1.set_xlim(param_var_range[0], param_var_range[-1] )
#
# ax2.plot( xtate0_list,xtate2_list, "b,")
# ax2.set_title("Bifurcation Diagram:y",fontsize=plt_sub_title_fontsize)
# # ax2.set_xlim(param_var_range[0], param_var_range[-1])
#
# ax3.plot( xtate0_list,xtate3_list, "g,")
# ax3.set_title("Bifurcation Diagram:z",fontsize=plt_sub_title_fontsize)
# # ax3.set_xlim(param_var_range[0], param_var_range[-1])
#
# ax4.plot( xtate1_list,xtate3_list, "y,")
# ax4.set_title("Bifurcation Diagram:u",fontsize=plt_sub_title_fontsize)
# # ax4.set_xlim(param_var_range[0], param_var_range[-1])
# # plt.xlabel("xlabel")
# # plt.ylabel("ylabel")
# plt.subplots_adjust(wspace=0.2, hspace=0.3)
# plt.suptitle("My Plot Title",fontsize=plt_title_fontsize)
# fig.subplots_adjust(left=0.05 ,right=0.95, bottom=0.07, top=0.88)
#
# plt.show()