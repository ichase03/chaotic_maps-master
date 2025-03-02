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
uc    = 0.5
# uc    = -0.
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
param_var_range_start = -1.5
param_var_range_end = 0.2
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
calc_num = 5000
plot_num_range = list(range(1, 101))

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

fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(14/2.54, 8/2.54))
ax1.plot( xtate0_list,xtate1_list, "r,")
# ax1.set_title("Diagram:x",fontsize=plt_sub_title_fontsize)
# ax1.set_xlim(param_var_range[0], param_var_range[-1] )
ax1.set_xlabel("$x_n$",labelpad=-2)
ax1.set_ylabel("$y_n$",labelpad=-1)

ax2.plot( xtate0_list,xtate2_list, "b,")
# ax2.set_title("Diagram:y",fontsize=plt_sub_title_fontsize)
# ax2.set_xlim(param_var_range[0], param_var_range[-1])
ax2.set_xlabel("$x_n$",labelpad=-1)
ax2.set_ylabel("$z_n$",labelpad=-10)

ax3.plot( xtate0_list,xtate3_list, "g,")
# ax3.set_title("Diagram:z",fontsize=plt_sub_title_fontsize)
# ax3.set_xlim(param_var_range[0], param_var_range[-1])
ax3.set_xlabel("$x_n$",labelpad=-1)
ax3.set_ylabel("$u_n$",labelpad=-1)

ax4.plot( xtate1_list,xtate3_list, "y,")
# ax4.set_title("Diagram:u",fontsize=plt_sub_title_fontsize)
ax4.set_xlabel("$y_n$",labelpad=-1)
ax4.set_ylabel("$z_n$",labelpad=-10)

# ax4.set_xlim(param_var_range[0], param_var_range[-1])
# plt.xlabel("xlabel")
# plt.ylabel("ylabel")
plt.subplots_adjust(wspace=0.15, hspace=0.40)
plt.suptitle("Phase diagram of attractor")
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.90)
plt.savefig('../pic/PIC_draw_5D_xiangtu_0.png',dpi=300)
plt.show()

fig, ((ax1, ax2),(ax3, ax4),(ax5,ax6)) = plt.subplots(3, 2, figsize=(14/2.54, 10/2.54))
ax1.plot( plot_num_range,xtate0_list[1:101], "r-")
# ax1.set_title("Diagram:x",fontsize=plt_sub_title_fontsize)
# ax1.set_xlim(param_var_range[0], param_var_range[-1] )
ax1.set_xlim(plot_num_range[0], plot_num_range[-1])
# ax1.set_xlabel("$x_n$",labelpad=-2)
ax1.set_xticks([])
ax1.set_ylabel("$x_n$",labelpad=-1)

ax2.plot( plot_num_range,xtate1_list[1:101], "b-")
# ax2.set_title("Diagram:y",fontsize=plt_sub_title_fontsize)
# ax2.set_xlim(param_var_range[0], param_var_range[-1])
ax2.set_xlim(plot_num_range[0], plot_num_range[-1])
# ax2.set_xlabel("$x_n$",labelpad=-1)
ax2.set_xticks([])
ax2.set_ylabel("$y_n$",labelpad=-5)

ax3.plot( plot_num_range,xtate2_list[1:101], "g-")
# ax3.set_title("Diagram:z",fontsize=plt_sub_title_fontsize)
ax3.set_xlim(plot_num_range[0], plot_num_range[-1])
# ax3.set_xlabel("$x_n$",labelpad=-1)
ax3.set_xticks([])
ax3.set_ylabel("$z_n$",labelpad=5)

ax4.plot( plot_num_range,xtate3_list[1:101], "y-")
# ax4.set_title("Diagram:u",fontsize=plt_sub_title_fontsize)
ax4.set_xlim(plot_num_range[0], plot_num_range[-1])
# ax4.set_xlabel("$y_n$",labelpad=-1)
ax4.set_xticks([])
ax4.set_ylabel("$u_n$",labelpad=5)

ax5.plot( plot_num_range,xtate4_list[1:101], "m-")
# ax4.set_title("Diagram:u",fontsize=plt_sub_title_fontsize)
ax5.set_xlim(plot_num_range[0], plot_num_range[-1])
# ax5.set_xlabel("$y_n$",labelpad=-1)
ax5.set_xticks([])
ax5.set_ylabel("$v_n$",labelpad=5)

ax6.axis('off')

# ax4.set_xlim(param_var_range[0], param_var_range[-1])
# plt.xlabel("xlabel")
# plt.ylabel("ylabel")
plt.subplots_adjust(wspace=0.25, hspace=0.10)
# plt.suptitle("Phase diagram of attractor")
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.01, top=0.99)
plt.savefig('../pic/PIC_draw_5D_xulie.png',dpi=300)
plt.show()

