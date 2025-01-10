import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm  # 导入 tqdm 库

# 返回在某点处的李雅普诺夫指数
def nD_chaotic_Lya_calc_fixpara( chaotic_map , jacobian , init , n_iterations , chaotic_dim_num ):
    x = init
    J = np.identity(chaotic_dim_num)
    lyapunov_sum = np.zeros(chaotic_dim_num)
    for i in range ( 2000 ):
        x = chaotic_map(x)

    for i in range(n_iterations):
        x = chaotic_map(x)
        J = jacobian(x) @ J  # 更新雅可比矩阵
        lyapunov_sum += np.log(np.abs(np.linalg.eigvals( J )))
    lyapunov = lyapunov_sum / (n_iterations + 2000)
    return lyapunov

# 返回在某点处的李雅普诺夫指数
# https://blog.csdn.net/qq_42665419/article/details/120401230
def nD_chaotic_Lya_calc_fixpara_new( chaotic_map , jacobian , init , n_iterations , chaotic_dim_num ):
    x = init
    Q = np.identity(chaotic_dim_num)
    lyapunov_sum = np.zeros(chaotic_dim_num)

    if (chaotic_dim_num == 1):
        for i in range(5000):
            x = chaotic_map(x)
        for i in range(n_iterations):
            x = chaotic_map(x)
            Jac = jacobian(x)
            # J = jacobian(x) @ J  # 更新雅可比矩阵
            lyapunov_sum += np.log(np.abs( Jac ) + 1e-16 )
    else :
        for i in range(5000):
            x = chaotic_map(x)
            Jac = jacobian(x)
            Q, _ = np.linalg.qr(Jac @ Q)  # Update Q during transient phase
        for i in range(n_iterations):
            x = chaotic_map(x)
            Jac = jacobian(x)
            # [Q,R] = np.linalg.qr( (Jac @ Q) )
            Q, R = scipy.linalg.qr( (Jac @ Q) , mode='economic' )
            # J = jacobian(x) @ J  # 更新雅可比矩阵
            lyapunov_sum += np.log( np.abs(np.diag( R )) + 1e-16 ) # Add a small constant to avoid log(0)
    lyapunov = lyapunov_sum / (n_iterations+5000)
    return lyapunov


def nD_chaotic_Lya_calc(
        chaotic_map     ,
        jacobian        ,
        init            ,
        n_iterations    ,
        chaotic_dim_num ,
        parameter_range ,
        parameter_seqnum
):
    para_range_len = len(parameter_range)
    # init_dim = len(init)
    lya_list = np.zeros([para_range_len, chaotic_dim_num])

    print("Running LE calc phase...")
    for i in tqdm(range(para_range_len), desc="Transient Phase"):
        init[parameter_seqnum] = parameter_range[i]
        lya_res = nD_chaotic_Lya_calc_fixpara_new ( chaotic_map , jacobian , init , n_iterations , chaotic_dim_num )
        lya_list[i] = lya_res

    return lya_list

def nD_lya_plot_dim4(
    chaotic_map , jacobian , init , n_iterations , chaotic_dim_num, parameter_range, parameter_seqnum
):
    lya_list = nD_chaotic_Lya_calc( chaotic_map , jacobian , init , n_iterations , chaotic_dim_num, parameter_range, parameter_seqnum )
    lya_list0 = lya_list[:,0]
    # lya_list1 = lya_list[:,1]
    # lya_list2 = lya_list[:,2]
    # lya_list3 = lya_list[:,3]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))

    # ax1.plot(y0_list, para_list, "k,", alpha=kwargs["alpha"])
    ax1.plot(parameter_range, lya_list0, "k-", linewidth=1)
    ax1.set_title("Bifurcation Diagram")
    ax1.set_xlim(parameter_range[0], parameter_range[-1])

    # ax2.plot(parameter_range, lya_list1, "b--")
    # ax2.set_title("Bifurcation Diagram")
    # ax2.set_xlim(parameter_range[0], parameter_range[-1])
    #
    # ax3.plot(parameter_range, lya_list2, "g--")
    # ax3.set_title("Bifurcation Diagram")
    # ax3.set_xlim(parameter_range[0], parameter_range[-1])
    #
    # ax4.plot(parameter_range, lya_list3, "y--")
    # ax4.set_title("Bifurcation Diagram")
    # ax4.set_xlim(parameter_range[0], parameter_range[-1])

    # if "xlabel" in kwargs:
    #     plt.xlabel(kwargs["xlabel"])
    #
    # if "ylabel" in kwargs:
    #     plt.ylabel(kwargs["ylabel"])

    plt.show()



# 返回在某参数取值处的分岔点阵列
def nD_bifurcation_data_fixed_x(
    f, init, y_points=100, dropped_steps=2000, **kwargs
):
    x = init

    for i in range(dropped_steps):
        x = f(x)
    x_dim = len(x)
    res = np.zeros([y_points,x_dim])

    for i in range(y_points):
        x = f(x)
        res[i] = x

    return res

def nD_bifurcation_data(
        f, init, parameter_range, parameter_seqnum, y_points=500, dropped_steps=1000, **kwargs
):
    para_range_len = len(parameter_range)
    init_dim = len(init)
    x_list = np.array([])
    y_list = np.zeros([para_range_len,y_points,init_dim])

    for i in range(para_range_len):
        init[parameter_seqnum] = parameter_range[i]
        y_res = nD_bifurcation_data_fixed_x( f, init, y_points=y_points, dropped_steps=dropped_steps, **kwargs)

        x_pad = np.full(y_points, parameter_range[i])
        x_list = np.append(x_list, x_pad)
        # y_list = np.vstack((y_list, y_res[None,...]))
        y_list[i] = y_res
    return x_list, y_list


def nD_bifurcation_plot_dim4(
    f, init, parameter_range, parameter_seqnum, y_points=500, dropped_steps=1000, **kwargs
):
    x_list, y_list = nD_bifurcation_data( f , init , parameter_range, parameter_seqnum, y_points, dropped_steps, **kwargs )
    para_list = y_list[:,:,parameter_seqnum].flatten()
    y0_list = y_list[:,:,0].flatten()
    # y1_list = y_list[:,:,1].flatten()
    # y2_list = y_list[:,:,2].flatten()
    # y3_list = y_list[:,:,3].flatten()
    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 9))

    # ax1.plot(y0_list, para_list, "k,", alpha=kwargs["alpha"])
    ax1.plot( para_list,y0_list, "r,")
    ax1.set_title("Bifurcation Diagram")
    ax1.set_xlim(parameter_range[0], parameter_range[-1])

    # ax2.plot( para_list,y1_list, "b,")
    # ax2.set_title("Bifurcation Diagram")
    # ax2.set_xlim(parameter_range[0], parameter_range[-1])
    #
    # ax3.plot( para_list,y2_list, "g,")
    # ax3.set_title("Bifurcation Diagram")
    # ax3.set_xlim(parameter_range[0], parameter_range[-1])
    #
    # ax4.plot( para_list,y3_list, "y,")
    # ax4.set_title("Bifurcation Diagram")
    # ax4.set_xlim(parameter_range[0], parameter_range[-1])

    # if "xlabel" in kwargs:
    #     plt.xlabel(kwargs["xlabel"])
    #
    # if "ylabel" in kwargs:
    #     plt.ylabel(kwargs["ylabel"])

    plt.show()











