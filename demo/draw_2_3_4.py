# 2_3_4 logistic pic

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

parameters_init = np.array((0.1,3.7))

# 相关控制参数
param_var_range_start = 2.5
param_var_range_end = 4
param_var_range = np.linspace(param_var_range_start,param_var_range_end,800)
state_plot_range = np.linspace(param_var_range_start,param_var_range_end,5)
param_seqnum = 1
y_points = 2000
dropped_steps = 2000
chaotic_dim_num = 1
n_iterations = 800

# xiangweitu huizhi
# 控制参数
calc_num = 100

model_param_dim = len(parameters_init)
state_list = np.zeros([calc_num,model_param_dim])
cur_state = parameters_init
for i in range(calc_num):
    # nxt_xtate = cm.sin_map_new( cur_state )
    nxt_xtate = cm.Logistic_map_new( cur_state )
    state_list[i] = nxt_xtate
    cur_state = nxt_xtate

xtate0_list = state_list[:,0].flatten()
# # 绘制混沌行为
# plt.plot(xtate0_list, 'b-')
# plt.xlabel('Iteration')
# plt.ylabel('x')
# plt.title(f'Logistic Map Chaos ( u = {parameters_init[1]})')
# plt.show()

# 计算关于参数parameters[parameter_seqnum]在param_var_range范围变动时的分岔点集
x_list, y_list = cm.nD_bifurcation_data(
    # f=cm.sin_map_new,
    f = cm.Logistic_map_new ,
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
    # chaotic_map      = cm.sin_map_new       ,
    # jacobian         = cm.sin_map_deriv_new ,
    chaotic_map      = cm.Logistic_map_new       ,
    jacobian         = cm.Logistic_map_deriv_new ,
    init             = parameters_init          ,
    n_iterations     = n_iterations             ,
    chaotic_dim_num  = chaotic_dim_num          ,
    parameter_range  = param_var_range          ,
    parameter_seqnum = param_seqnum
)

lya_list0 = lya_list[:,0]

# 绘图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15/2.54, 5/2.54))
ax1.plot( xtate0_list, 'b-', linewidth=0.5)
# ax1.set_title(f'Sine Map Chaos ( u = 0.75 )',fontsize=plt_sub_title_fontsize, fontweight='bold')
# ax1.set_title(f'Sine Map Chaos ( u = 0.75 )')
# ax1.set_title(f'Logistic Map Chaos ( u = {parameters_init[1]})')
ax1.set_title(f'Logistic Map Chaos ( u = 3.7 )')
ax1.set_xlabel('Iteration',fontsize=plt_sub_title_fontsize)
ax1.set_ylabel('x',fontsize=plt_sub_title_fontsize)
# 调整刻度标签字体大小
ax1.tick_params(axis='both', labelsize=plt_sub_title_fontsize)


ax2.plot( para_list,y0_list, "r,")
# ax2.set_title("Bifurcation Diagram",fontsize=plt_sub_title_fontsize, fontweight='bold')
ax2.set_title("Bifurcation Diagram")
ax2.set_xlabel('parameter:u',fontsize=plt_sub_title_fontsize)
ax2.set_ylabel('x_bifurcation',fontsize=plt_sub_title_fontsize)
ax2.set_xlim(param_var_range[0], param_var_range[-1] )
ax2.tick_params(axis='both', labelsize=plt_sub_title_fontsize)

ax3.plot(param_var_range, lya_list0, "b-", linewidth=1, label = 'LE1')
ax3.plot(param_var_range, param_var_range*0, "k--", linewidth=0.5 )
# ax3.set_title("LE Diagram",fontsize=plt_sub_title_fontsize, fontweight='bold')
ax3.set_title("LE Diagram")
ax3.set_xlabel('parameter:u',fontsize=plt_sub_title_fontsize)
ax3.set_ylabel('LE',fontsize=plt_sub_title_fontsize)
ax3.set_xlim(param_var_range[0], param_var_range[-1])
ax3.tick_params(axis='both', labelsize=plt_sub_title_fontsize)

# plt.xlabel("xlabel")
# plt.ylabel("ylabel")
# plt.subplots_adjust(wspace=0.2, hspace=0.3)
# plt.suptitle("My Plot Title",fontsize=plt_title_fontsize)
fig.subplots_adjust(left=0.08 ,right=0.98, bottom=0.22, top=0.88,wspace=0.42)
# fig.subplots_adjust(left=0.05 ,right=0.98)
# plt.tight_layout()
plt.savefig("../pic/PIC_2_3_4_LOGISTIC.png", dpi=300)
plt.show()