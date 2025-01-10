import numpy as np
import matplotlib.pyplot as plt
import chaotic_maps as cm

sin_generator = cm.sin_map_family(0.9)
sin_dynamic = sin_generator(.8)

henon_generator = cm.henon_map_family(0.3)
henon_dynamic = henon_generator(1.4)

ux1 = 0.1
ux2 = 0.1
ux3 = 0.1
ux4 = 0.1
xstate0 = [ux1,ux2,ux3,ux4]
#  模型已知的参数
ua    = 0.5
ub    = 0.5
uc    = 0.5
ud    = 0.5
ue    = 0.5
uk0   = 0.5
uk1   = 0.5
utao0 = 0.5
utao1 = 0.5
parameters = [ua,ub,uc,ud,ue,uk0,uk1,utao0,utao1]

parameters_init = np.array(( ux1,ux2,ux3,ux4,ua,ub,uc,ud,ue,uk0,uk1,utao0,utao1 ))
#
# Lya = cm.nD_chaotic_Lya_calc( cm.nD_chaotic_func , cm.nD_chaotic_func_Jac , parameters_init , 10 , 4 )
# print(Lya)
#
# x_list, y_list = cm.nD_bifurcation_data( cm.nD_chaotic_func , parameters_init, np.linspace(0.7,1.8,100), 4  )
# print(x_list)
# print(y_list)

# cm.nD_bifurcation_plot_dim4( cm.nD_chaotic_func , parameters_init, np.linspace(0.7,1.8,200), 4  )
# cm.nD_lya_plot_dim4( cm.nD_chaotic_func , cm.nD_chaotic_func_Jac , parameters_init , n_iterations = 100 , chaotic_dim_num = 4 , parameter_range = np.linspace(0.7,0.9,180) , parameter_seqnum = 4)
cm.nD_lya_plot_dim4(
    cm.sin_map_new ,
    cm.sin_map_deriv_new ,
    np.array((.2,.8)) ,
    n_iterations = 200 ,
    chaotic_dim_num = 1 ,
    parameter_range = np.linspace(0.6,1,1000) ,
    parameter_seqnum = 1
)

cm.nD_bifurcation_plot_dim4(
    cm.sin_map_new ,
    np.array((.2,.8)) ,
    parameter_range = np.linspace(0.6,1,1000) ,
    parameter_seqnum = 1
)

# fig, axes = plt.subplots(2, 2)
# cm.cobweb_plot_nD( cm.nD_chaotic_func, parameters_init , 0, 20, domain=np.linspace(-0.7, 1.3), ax=axes[0][0],ylabel=r'$x_{n+1}$')
# # cm.cobweb_plot_nD( henon_dynamic, np.array([0.1,0.1]), 0, 20 , domain=np.linspace(-0.7, 1.3), ax=axes[0][0], ylabel=r'$x_{n+1}$')
# # cm.cobweb_plot(cm.sin_map(.7), .1, 20, domain=np.linspace(0, 1), ax=axes[0][0], ylabel=r'$x_{n+1}$')
# cm.cobweb_plot(cm.sin_map(.8), .1, 20, domain=np.linspace(0, 1), ax=axes[0][1])
# cm.cobweb_plot(cm.sin_map(.85), .1, 20, domain=np.linspace(0, 1), ax=axes[1][0], xlabel=r'$x_n$', ylabel=r'$x_{n+1}$')
# cm.cobweb_plot(cm.sin_map(.9), .1, 20, domain=np.linspace(0, 1), ax=axes[1][1], xlabel=r'$x_n$')
# axes[0][0].set_title(r'$\mu = 0.7$')
# axes[0][1].set_title(r'$\mu = 0.8$')
# axes[1][0].set_title(r'$\mu = 0.85$')
# axes[1][1].set_title(r'$\mu = 0.9$')
# plt.tight_layout()
# plt.show()
# # plt.savefig("sin_cobwebs.png", dpi=300)
#
#
#
# cm.bifurcation_and_lyapunov_plots(
#    cm.sin_map,
#    init=0.2,
#    parameter_range=np.linspace(0.6, 1, num=1000),
#    deriv_generator=cm.sin_map_deriv,
#    y_points=200,
#    xlabel=r"$\mu$",
#    ylabel=r"$x$"
#    # file_name="sine.png",
# )

# fig, axes = plt.subplots(1,2, figsize=(12,5))
#
# cm.bifurcation_plot(
#     cm.gauss_map_family(alpha=5),
#     init=0.1,
#     parameter_range=np.linspace(-0.9, 1, 10000),
#     y_points=500,
#     xlabel=r"$\beta$",
#     ylabel=r"$x$",
#     set_title=r"$\alpha = 5$",
#     ax=axes[0]
# )
#
# cm.bifurcation_plot(
#     cm.gauss_map_family(alpha=6.5),
#     init=0.1,
#     parameter_range=np.linspace(-0.9, 1, 10000), #10000
#     y_points=500,
#     xlabel=r"$\beta$",
#     ylabel=r"$x$",
#     set_title=r"$\alpha = 6.5$",
#     ax=axes[1]
# )
#
# plt.savefig("gauss.png", dpi=300)
