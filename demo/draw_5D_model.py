import numpy as np
import matplotlib.pyplot as plt
import chaotic_maps as cm

plt_title_fontsize = 10
plt_sub_title_fontsize = 10
# 全局设置字体为新罗马
plt.rcParams["font.family"] = "Times New Roman"
# 全局设置字体和文字大小
plt.rcParams["font.size"] = 10  # 默认字体大小
plt.rcParams["axes.titlesize"] = 10  # 标题字体大小
plt.rcParams["axes.labelsize"] = 10  # 坐标轴标签字体大小
plt.rcParams["xtick.labelsize"] = 10  # X轴刻度标签字体大小
plt.rcParams["ytick.labelsize"] = 10  # Y轴刻度标签字体大小
plt.rcParams["legend.fontsize"] = 10  # 图例字体大小

#%%
# 模型基础参数设置
ux1 = 0.5
ux2 = 0.5
ux3 = 0.1
ux4 = 0.1
ux5 = 0.1
# xstate0 = [ux1,ux2,ux3,ux4]
#  模型已知的参数
ua    = 0.7
ub    = 0.1
uc    = -0.025
# uc    = -0.025
ud    = -1.5
ue    = np.pi/6
uf    = -1
uk0   = 1
uk1   = -10
utao0 = 1
utao1 = 0.5
utao2 = 0.5
# parameters = [ua,ub,uc,ud,ue,uk0,uk1,utao0,utao1]
parameters_init = np.array(( ux1,ux2,ux3,ux4,ux5, ua,ub,uc,ud,ue,uf, uk0,uk1,utao0,utao1, utao2 ))

# 相关控制参数
param_var_range_start = 0.2
param_var_range_end = 1.5
param_var_range = np.linspace(param_var_range_start,param_var_range_end,600)
state_plot_range = np.linspace(param_var_range_start,param_var_range_end,5)
param_seqnum = 7
y_points = 200
dropped_steps = 2000
chaotic_dim_num = 5
n_iterations = 2000
cowbo_iterations = 50

# xiangweitu huizhi
# 控制参数
calc_num = 50000

model_param_dim = len(parameters_init)
state_list = np.zeros([calc_num,model_param_dim])
cur_state = parameters_init
for i in range(calc_num):
    nxt_xtate = cm.mod_5D_chaotic_func( cur_state )
    state_list[i] = nxt_xtate
    cur_state = nxt_xtate

xtate0_list = state_list[:,0].flatten()
xtate1_list = state_list[:,1].flatten()
xtate2_list = state_list[:,2].flatten()
xtate3_list = state_list[:,3].flatten()
xtate4_list = state_list[:,4].flatten()

state_01_xn_list = xtate0_list
state_02_yn_list = xtate1_list
state_03_zn_list = xtate2_list
state_04_un_list = xtate3_list
state_05_vn_list = xtate4_list

state = np.vstack(( xtate0_list ,
                    xtate1_list ,
                    xtate2_list ,
                    xtate3_list ,
                    xtate4_list ))

fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(15/2.54, 9/2.54))
ax1.plot( xtate0_list,xtate1_list, "r,")
# ax1.set_title("Diagram:x",fontsize=plt_sub_title_fontsize)
# ax1.set_xlim(param_var_range[0], param_var_range[-1] )
ax1.set_xlabel("$x_n$",labelpad=-2)
ax1.set_ylabel("$y_n$",labelpad=-1)

ax2.plot( xtate0_list,xtate2_list, "b,")
# ax2.set_title("Diagram:y",fontsize=plt_sub_title_fontsize)
# ax2.set_xlim(param_var_range[0], param_var_range[-1])
ax2.set_xlabel("$x_n$",labelpad=-1)
ax2.set_ylabel("$z_n$",labelpad=-1)

ax3.plot( xtate0_list,xtate3_list, "g,")
# ax3.set_title("Diagram:z",fontsize=plt_sub_title_fontsize)
# ax3.set_xlim(param_var_range[0], param_var_range[-1])
ax3.set_xlabel("$x_n$",labelpad=-1)
ax3.set_ylabel("$u_n$",labelpad=-1)

ax4.plot( xtate1_list,xtate3_list, "y,")
# ax4.set_title("Diagram:u",fontsize=plt_sub_title_fontsize)
ax4.set_xlabel("$y_n$",labelpad=-1)
ax4.set_ylabel("$z_n$",labelpad=-1)

# ax4.set_xlim(param_var_range[0], param_var_range[-1])
# plt.xlabel("xlabel")
# plt.ylabel("ylabel")
plt.subplots_adjust(wspace=0.15, hspace=0.32)
plt.suptitle("Phase diagram of attractor")
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.90)
plt.savefig('../pic/PIC_draw_5D_phase_0.png',dpi=300)
plt.show()

# fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(15/2.54, 7/2.54))
# for i in range(cowbo_iterations):
#     ax1.plot([xtate0_list[i], xtate0_list[i]]  , [xtate0_list[i]  , xtate0_list[i+1]], "b-")
#     ax1.plot([xtate0_list[i], xtate0_list[i+1]], [xtate0_list[i+1], xtate0_list[i+1]], "b-")
#     ax1.plot( xtate0_list[i], xtate0_list[i+1], "ok")
#     ax1.set_xlabel("$x_n$", labelpad=-2)
#     ax1.set_ylabel("$x_{n+1}$", labelpad=-2)
#
#     ax2.plot([xtate1_list[i], xtate1_list[i]]  , [xtate1_list[i]  , xtate1_list[i+1]], "b-")
#     ax2.plot([xtate1_list[i], xtate1_list[i+1]], [xtate1_list[i+1], xtate1_list[i+1]], "b-")
#     ax2.plot( xtate1_list[i], xtate1_list[i+1], "ok")
#     ax2.set_xlabel("$y_n$", labelpad=-2)
#     ax2.set_ylabel("$y_{n+1}$", labelpad=-2)
#
#     ax3.plot([xtate2_list[i], xtate2_list[i]]  , [xtate2_list[i]  , xtate2_list[i+1]], "b-")
#     ax3.plot([xtate2_list[i], xtate2_list[i+1]], [xtate2_list[i+1], xtate2_list[i+1]], "b-")
#     ax3.plot( xtate2_list[i], xtate2_list[i+1], "ok")
#     ax3.set_xlabel("$z_n$", labelpad=-2)
#     ax3.set_ylabel("$z_{n+1}$", labelpad=-1)
#
#     ax4.plot([xtate3_list[i], xtate3_list[i]]  , [xtate3_list[i]  , xtate3_list[i+1]], "b-")
#     ax4.plot([xtate3_list[i], xtate3_list[i+1]], [xtate3_list[i+1], xtate3_list[i+1]], "b-")
#     ax4.plot( xtate3_list[i], xtate3_list[i+1], "ok")
#     ax4.set_xlabel("$u_n$", labelpad=-2)
#     ax4.set_ylabel("$u_{n+1}$", labelpad=-1)
# plt.subplots_adjust(wspace=0.15, hspace=0.32)
# plt.suptitle("Cobwebs draw")
# fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.90)
# plt.savefig('../pic/PIC_draw_5D_phase_cobwebs.png',dpi=300)
# plt.show()



for k in range(len(state_plot_range)):
    model_param_dim = len(parameters_init)
    state_list = np.zeros([calc_num,model_param_dim])
    cur_state = parameters_init
    cur_state[param_seqnum] =  state_plot_range[k]
    for i in range(calc_num):
        nxt_xtate = cm.mod_5D_chaotic_func( cur_state )
        state_list[i] = nxt_xtate
        cur_state = nxt_xtate

    xtate0_list = state_list[:,0].flatten()
    xtate1_list = state_list[:,1].flatten()
    xtate2_list = state_list[:,2].flatten()
    xtate3_list = state_list[:,3].flatten()

    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(15/2.54, 9/2.54))
    ax1.plot( xtate0_list,xtate1_list, "r,")
    # ax1.set_title("Diagram:x",fontsize=plt_sub_title_fontsize)
    # ax1.set_xlim(param_var_range[0], param_var_range[-1] )
    ax1.set_xlabel("x",labelpad=-2)
    ax1.set_ylabel("y",labelpad=-1)


    ax2.plot( xtate0_list,xtate2_list, "b,")
    # ax2.set_title("Diagram:y",fontsize=plt_sub_title_fontsize)
    # ax2.set_xlim(param_var_range[0], param_var_range[-1])
    ax2.set_xlabel("x",labelpad=-1)
    ax2.set_ylabel("z",labelpad=-1)

    ax3.plot( xtate0_list,xtate3_list, "g,")
    # ax3.set_title("Diagram:z",fontsize=plt_sub_title_fontsize)
    # ax3.set_xlim(param_var_range[0], param_var_range[-1])
    ax3.set_xlabel("x",labelpad=-1)
    ax3.set_ylabel("u",labelpad=-1)

    ax4.plot( xtate1_list,xtate3_list, "y,")
    # ax4.set_title("Diagram:u",fontsize=plt_sub_title_fontsize)
    ax4.set_xlabel("y",labelpad=-1)
    ax4.set_ylabel("z",labelpad=-1)
    # ax4.set_xlim(param_var_range[0], param_var_range[-1])
    # plt.xlabel("xlabel")
    # plt.ylabel("ylabel")
    plt.subplots_adjust(wspace=0.15 , hspace=0.32)
    plt.suptitle("Phase diagram of attractor")
    fig.subplots_adjust(left=0.08 ,right=0.98, bottom=0.12, top=0.90)
    # plt.tight_layout()
    filename = f'../pic/PIC_draw_5D_phase_{k + 1}.png'
    plt.savefig(filename, dpi=300)
    plt.show()

#%%
# 分岔点图绘制
# 计算关于参数parameters[parameter_seqnum]在param_var_range范围变动时的分岔点集
x_list, y_list = cm.nD_bifurcation_data(
    f = cm.mod_5D_chaotic_func              ,
    init = parameters_init              ,
    parameter_range = param_var_range   ,
    parameter_seqnum = param_seqnum ,
    y_points = y_points,
    dropped_steps = dropped_steps
)

# 将计算结果展开为序列
para_list = y_list[:,:,param_seqnum].flatten()
y0_list = y_list[:,:,0].flatten()
y1_list = y_list[:,:,1].flatten()
y2_list = y_list[:,:,2].flatten()
y3_list = y_list[:,:,3].flatten()
y5_list = y_list[:,:,4].flatten()

# 绘制分岔图
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(15/2.54, 9/2.54))
ax1.plot( para_list,y0_list, "r,")
# ax1.set_title("Bifurcation Diagram:x",fontsize=plt_sub_title_fontsize)
ax1.set_xlim(param_var_range[0], param_var_range[-1] )
# ax1.set_xlabel("y", labelpad=-2)
ax1.set_ylabel("$x_n$", labelpad=-2)

ax2.plot( para_list,y1_list, "b,")
# ax2.set_title("Bifurcation Diagram:y",fontsize=plt_sub_title_fontsize)
ax2.set_xlim(param_var_range[0], param_var_range[-1])
ax2.set_ylabel("$y_n$", labelpad=-2)

ax3.plot( para_list,y2_list, "g,")
# ax3.set_title("Bifurcation Diagram:z",fontsize=plt_sub_title_fontsize)
ax3.set_xlim(param_var_range[0], param_var_range[-1])
ax3.set_xlabel("c", labelpad=-2)
ax3.set_ylabel("$z_n$", labelpad=0)

ax4.plot( para_list,y3_list, "y,")
# ax4.set_title("Bifurcation Diagram:u",fontsize=plt_sub_title_fontsize)
ax4.set_xlim(param_var_range[0], param_var_range[-1])
ax4.set_xlabel("c", labelpad=-2)
ax4.set_ylabel("$u_n$", labelpad=0)

# plt.xlabel("xlabel")
# plt.ylabel("ylabel")
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.suptitle("Bifurcation Diagram")
fig.subplots_adjust(left=0.07 ,right=0.98, bottom=0.12, top=0.90)
plt.savefig('../pic/PIC_draw_5D_Bif', dpi=300)
plt.show()


lya_list = cm.nD_chaotic_Lya_calc_multiprocessing(
    chaotic_map      = cm.mod_5D_chaotic_func          ,
    jacobian         = cm.mod_5D_chaotic_func_Jac      ,
    init             = parameters_init          ,
    n_iterations     = n_iterations             ,
    chaotic_dim_num  = chaotic_dim_num          ,
    parameter_range  = param_var_range          ,
    parameter_seqnum = param_seqnum
)

# lya_list0 = lya_list[:,0]
# lya_list1 = lya_list[:,1]
# lya_list2 = lya_list[:,2]
# lya_list3 = lya_list[:,3]
lya_list0 = [arr[0] for arr in lya_list]
lya_list1 = [arr[1] for arr in lya_list]
lya_list2 = [arr[2] for arr in lya_list]
lya_list3 = [arr[3] for arr in lya_list]
lya_list4 = [arr[4] for arr in lya_list]

fig, ( ax1 ) = plt.subplots(1, 1, figsize=(15/2.54, 6/2.54))


l0, = ax1.plot(param_var_range, lya_list0, "r-", linewidth=1, label = 'LE1')
ax1.set_xlim(param_var_range[0], param_var_range[-1])
l1, = ax1.plot(param_var_range, lya_list1, "b-", label = 'LE2')
l2, = ax1.plot(param_var_range, lya_list2, "g-", label = 'LE3')
l3, = ax1.plot(param_var_range, lya_list3, "y-", label = 'LE4')
l4, = ax1.plot(param_var_range, lya_list4, "c-", label = 'LE5')

ax1.legend( handles=[l0,l1,l2,l3,l4],labels = [ 'LE0', 'LE1', 'LE2', 'LE3', 'LE4' ], loc = 'upper left', fontsize = plt_sub_title_fontsize, ncol=2,bbox_to_anchor=(0.65, 0.85) )
plt.xlabel("param:c", labelpad=-3)
plt.ylabel("Lya", labelpad=-3)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.suptitle("Lya diagram",fontsize=plt_title_fontsize)
fig.subplots_adjust(left=0.08 ,right=0.98, bottom=0.16, top=0.91)
plt.savefig('../pic/PIC_draw_5D_Lya', dpi=300)
plt.show()