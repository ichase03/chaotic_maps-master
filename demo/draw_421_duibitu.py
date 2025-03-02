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
calc_num = 40000
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

FPGA_calc_num = 20000
fpga_noise = 0.001
FPGA_parameters_init = np.array(( ux1+fpga_noise,ux2+fpga_noise,ux3+fpga_noise,ux4+fpga_noise,ux5+fpga_noise, ua,ub,uc,ud,ue,uf, uk0,uk1,utao0,utao1, utao2 ))
FPGA_state_list = np.zeros([FPGA_calc_num,model_param_dim])
FPGA_cur_state = FPGA_parameters_init
for i in range(FPGA_calc_num):
    FPGA_nxt_xtate = cm.mod_5D_chaotic_func( FPGA_cur_state )
    FPGA_state_list[i] = FPGA_nxt_xtate
    FPGA_cur_state = FPGA_nxt_xtate

FPGA_xtate0_list = FPGA_state_list[:,0].flatten()
FPGA_xtate1_list = FPGA_state_list[:,1].flatten()
FPGA_xtate2_list = FPGA_state_list[:,2].flatten()
FPGA_xtate3_list = FPGA_state_list[:,3].flatten()
FPGA_xtate4_list = FPGA_state_list[:,4].flatten()

fig, ( ax1 ) = plt.subplots(1, 1, figsize=(7/2.54, 4/2.54))
plt1,= ax1.plot(xtate0_list, xtate1_list, "r,")
plt2, = ax1.plot(FPGA_xtate0_list, FPGA_xtate1_list, "b,")
plt3,= ax1.plot(xtate0_list[0], xtate1_list[0], "r")
plt4, = ax1.plot(FPGA_xtate0_list[0], FPGA_xtate1_list[0], "b")
ax1.set_xlabel("$x_n$",labelpad=1)
ax1.set_ylabel("$y_n$",labelpad=1)
ax1.legend( handles=[plt3,plt4],labels = [ 'Python', 'FPGA' ], loc = 'upper right' )
# legend = ax1.legend()
# for handle in legend.legend_handles:
#     handle.set_markersize(10)
fig.subplots_adjust(left=0.20, right=0.95, bottom=0.23, top=0.95)
plt.savefig('../txt/PIC_draw_xn_yn_DUIBITU.png',dpi=300)
plt.show()

fig, ( ax1 ) = plt.subplots(1, 1, figsize=(7/2.54, 4/2.54))
plt1,= ax1.plot(xtate0_list, xtate2_list, "r,")
plt2, = ax1.plot(FPGA_xtate0_list, FPGA_xtate2_list, "b,")
plt3,= ax1.plot(xtate0_list[0], xtate2_list[0], "r")
plt4, = ax1.plot(FPGA_xtate0_list[0], FPGA_xtate2_list[0], "b")
ax1.set_xlabel("$x_n$",labelpad=1)
ax1.set_ylabel("$z_n$",labelpad=1)
ax1.legend( handles=[plt3,plt4],labels = [ 'Python', 'FPGA' ], loc = 'upper left' )
# legend = ax1.legend()
# for handle in legend.legend_handles:
#     handle.set_markersize(10)
fig.subplots_adjust(left=0.20, right=0.95, bottom=0.23, top=0.95)
plt.savefig('../txt/PIC_draw_xn_zn_DUIBITU.png',dpi=300)
plt.show()

fig, ( ax1 ) = plt.subplots(1, 1, figsize=(7/2.54, 4/2.54))
plt1,= ax1.plot(xtate0_list, xtate3_list, "r,")
plt2, = ax1.plot(FPGA_xtate0_list, FPGA_xtate3_list, "b,")
plt3,= ax1.plot(xtate0_list[0], xtate3_list[0], "r")
plt4, = ax1.plot(FPGA_xtate0_list[0], FPGA_xtate3_list[0], "b")
ax1.set_xlabel("$x_n$",labelpad=1)
ax1.set_ylabel("$u_n$",labelpad=1)
ax1.legend( handles=[plt3,plt4],labels = [ 'Python', 'FPGA' ], loc = 'upper left' )
# legend = ax1.legend()
# for handle in legend.legend_handles:
#     handle.set_markersize(10)
fig.subplots_adjust(left=0.20, right=0.95, bottom=0.23, top=0.95)
plt.savefig('../txt/PIC_draw_xn_un_DUIBITU.png',dpi=300)
plt.show()

fig, ( ax1 ) = plt.subplots(1, 1, figsize=(7/2.54, 4/2.54))
plt1,= ax1.plot(xtate1_list, xtate2_list, "r,")
plt2, = ax1.plot(FPGA_xtate1_list, FPGA_xtate2_list, "b,")
plt3,= ax1.plot(xtate1_list[0], xtate2_list[0], "r")
plt4, = ax1.plot(FPGA_xtate1_list[0], FPGA_xtate2_list[0], "b")
ax1.set_xlabel("$y_n$",labelpad=1)
ax1.set_ylabel("$z_n$",labelpad=1)
ax1.legend( handles=[plt3,plt4],labels = [ 'Python', 'FPGA' ], loc = 'upper right' )
# legend = ax1.legend()
# for handle in legend.legend_handles:
#     handle.set_markersize(10)
fig.subplots_adjust(left=0.20, right=0.95, bottom=0.23, top=0.95)
plt.savefig('../txt/PIC_draw_yn_zn_DUIBITU.png',dpi=300)
plt.show()