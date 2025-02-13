import numpy as np
import matplotlib.pyplot as plt
import chaotic_maps as cm

plt_title_fontsize = 20
plt_sub_title_fontsize = 18

#%%
# 模型基础参数设置
ux1 = 0.5
ux2 = 0.5
ux3 = 0.1
ux4 = 0.1
# xstate0 = [ux1,ux2,ux3,ux4]
#  模型已知的参数
ua    = 1.2
ub    = 0.1
uc    = 0.5
ud    = 1.72
ue    = np.pi/10
uk0   = 0.1
uk1   = -5
utao0 = 0.5
utao1 = 0.5
# parameters = [ua,ub,uc,ud,ue,uk0,uk1,utao0,utao1]
parameters_init = np.array(( ux1,ux2,ux3,ux4,ua,ub,uc,ud,ue,uk0,uk1,utao0,utao1 ))

# 相关控制参数
param_var_range_start = -2.5
param_var_range_end = 2.5
param_var_range = np.linspace(param_var_range_start,param_var_range_end,300)
state_plot_range = np.linspace(param_var_range_start,param_var_range_end,5)
param_seqnum = 4
y_points = 200
dropped_steps = 2000
chaotic_dim_num = 4
n_iterations = 8000

# xiangweitu huizhi
# 控制参数
calc_num = 200000

model_param_dim = len(parameters_init)
state_list = np.zeros([calc_num,model_param_dim])
cur_state = parameters_init
for i in range(calc_num):
    nxt_xtate = cm.nD_chaotic_func( cur_state )
    state_list[i] = nxt_xtate
    cur_state = nxt_xtate

xtate0_list = state_list[:,0].flatten()
xtate1_list = state_list[:,1].flatten()
xtate2_list = state_list[:,2].flatten()
xtate3_list = state_list[:,3].flatten()

fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
ax1.plot( xtate0_list,xtate1_list, "r,")
ax1.set_title("Diagram:x",fontsize=plt_sub_title_fontsize)
# ax1.set_xlim(param_var_range[0], param_var_range[-1] )

ax2.plot( xtate0_list,xtate2_list, "b,")
ax2.set_title("Diagram:y",fontsize=plt_sub_title_fontsize)
# ax2.set_xlim(param_var_range[0], param_var_range[-1])

ax3.plot( xtate0_list,xtate3_list, "g,")
ax3.set_title("Diagram:z",fontsize=plt_sub_title_fontsize)
# ax3.set_xlim(param_var_range[0], param_var_range[-1])

ax4.plot( xtate1_list,xtate3_list, "y,")
ax4.set_title("Diagram:u",fontsize=plt_sub_title_fontsize)
# ax4.set_xlim(param_var_range[0], param_var_range[-1])
# plt.xlabel("xlabel")
# plt.ylabel("ylabel")
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.suptitle("fixed_point draw",fontsize=plt_title_fontsize)
fig.subplots_adjust(left=0.05 ,right=0.95, bottom=0.07, top=0.88)

plt.show()

for k in range(len(state_plot_range)):
    model_param_dim = len(parameters_init)
    state_list = np.zeros([calc_num,model_param_dim])
    cur_state = parameters_init
    cur_state[param_seqnum] =  state_plot_range[k]
    for i in range(calc_num):
        nxt_xtate = cm.nD_chaotic_func( cur_state )
        state_list[i] = nxt_xtate
        cur_state = nxt_xtate

    xtate0_list = state_list[:,0].flatten()
    xtate1_list = state_list[:,1].flatten()
    xtate2_list = state_list[:,2].flatten()
    xtate3_list = state_list[:,3].flatten()

    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
    ax1.plot( xtate0_list,xtate1_list, "r,")
    ax1.set_title("Diagram:x",fontsize=plt_sub_title_fontsize)
    # ax1.set_xlim(param_var_range[0], param_var_range[-1] )

    ax2.plot( xtate0_list,xtate2_list, "b,")
    ax2.set_title("Diagram:y",fontsize=plt_sub_title_fontsize)
    # ax2.set_xlim(param_var_range[0], param_var_range[-1])

    ax3.plot( xtate0_list,xtate3_list, "g,")
    ax3.set_title("Diagram:z",fontsize=plt_sub_title_fontsize)
    # ax3.set_xlim(param_var_range[0], param_var_range[-1])

    ax4.plot( xtate1_list,xtate3_list, "y,")
    ax4.set_title("Diagram:u",fontsize=plt_sub_title_fontsize)
    # ax4.set_xlim(param_var_range[0], param_var_range[-1])
    # plt.xlabel("xlabel")
    # plt.ylabel("ylabel")
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.suptitle("My Plot Title",fontsize=plt_title_fontsize)
    fig.subplots_adjust(left=0.05 ,right=0.95, bottom=0.07, top=0.88)

    plt.show()

#%%
# 分岔点图绘制
# 计算关于参数parameters[parameter_seqnum]在param_var_range范围变动时的分岔点集
x_list, y_list = cm.nD_bifurcation_data(
    f = cm.nD_chaotic_func              ,
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

# 绘制分岔图
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
ax1.plot( para_list,y0_list, "r,")
ax1.set_title("Bifurcation Diagram:x",fontsize=plt_sub_title_fontsize)
ax1.set_xlim(param_var_range[0], param_var_range[-1] )

ax2.plot( para_list,y1_list, "b,")
ax2.set_title("Bifurcation Diagram:y",fontsize=plt_sub_title_fontsize)
ax2.set_xlim(param_var_range[0], param_var_range[-1])

ax3.plot( para_list,y2_list, "g,")
ax3.set_title("Bifurcation Diagram:z",fontsize=plt_sub_title_fontsize)
ax3.set_xlim(param_var_range[0], param_var_range[-1])

ax4.plot( para_list,y3_list, "y,")
ax4.set_title("Bifurcation Diagram:u",fontsize=plt_sub_title_fontsize)
ax4.set_xlim(param_var_range[0], param_var_range[-1])
# plt.xlabel("xlabel")
# plt.ylabel("ylabel")
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.suptitle("My Plot Title",fontsize=plt_title_fontsize)
fig.subplots_adjust(left=0.05 ,right=0.95, bottom=0.07, top=0.88)

plt.show()


lya_list = cm.nD_chaotic_Lya_calc_multiprocessing(
    chaotic_map      = cm.nD_chaotic_func          ,
    jacobian         = cm.nD_chaotic_func_Jac      ,
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

fig, ( ax1 ) = plt.subplots(1, 1, figsize=(9, 6))


l0, = ax1.plot(param_var_range, lya_list0, "r-", linewidth=1, label = 'LE1')
ax1.set_xlim(param_var_range[0], param_var_range[-1])
l1, = ax1.plot(param_var_range, lya_list1, "b-", label = 'LE1')
l2, = ax1.plot(param_var_range, lya_list2, "g-", label = 'LE1')
l3, = ax1.plot(param_var_range, lya_list3, "y-", label = 'LE1')

ax1.legend( handles=[l0,l1,l2,l3],labels = [ 'LE0', 'LE1', 'LE2', 'LE3' ], loc = 'best', fontsize = plt_sub_title_fontsize )
plt.xlabel("param",fontsize = plt_sub_title_fontsize)
plt.ylabel("Lya",fontsize = plt_sub_title_fontsize)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.suptitle("Lya diagram",fontsize=plt_title_fontsize)
fig.subplots_adjust(left=0.10 ,right=0.96, bottom=0.11, top=0.91)

plt.show()