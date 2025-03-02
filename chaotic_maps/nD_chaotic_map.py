import numpy as np
import math

# from demo.demo import parameters


# def logistic_map(r):
#     def dynamic(x):
#         return r * x * (1 - x)
#
#     return dynamic

def nD_chaotic_func( parameters ):
    # 根据不同的模型输入不同的数据
    ux1    = parameters [0]
    ux2    = parameters [1]
    ux3    = parameters [2]
    ux4    = parameters [3]
    # 模型已知的参数
    ua     = parameters [4]
    ub     = parameters [5]
    uc     = parameters [6]
    ud     = parameters [7]
    ue     = parameters [8]
    uk0    = parameters [9]
    uk1    = parameters [10]
    utao0  = parameters [11]
    utao1  = parameters [12]

    # 混沌系统
    ux1n = ua*ux2 + uc * math.sin( ue*ux1*( uk0*ux3**2 + uk1*ux3**4 ))
    ux2n = ub*ux1 + ud*ux2 * math.cos(ux4)
    # ux2n = ub*ux1 + ud * math.cos( ux2*( uk0*ux4**2 + uk1*ux4**4 ) )
    ux3n = ux3 + utao0 * ux2
    ux4n = ux4 + utao1 * ux2

    # 返回迭代值
    parameters_n = np.array(( ux1n, ux2n, ux3n, ux4n, ua, ub, uc, ud, ue, uk0, uk1, utao0, utao1 ))
    return parameters_n

def nD_chaotic_func_Jac ( parameters ):
    # 根据不同的模型输入不同的数据
    ux1    = parameters [0]
    ux2    = parameters [1]
    ux3    = parameters [2]
    ux4    = parameters [3]
    # 模型已知的参数
    ua     = parameters [4]
    ub     = parameters [5]
    uc     = parameters [6]
    ud     = parameters [7]
    ue     = parameters [8]
    uk0    = parameters [9]
    uk1    = parameters [10]
    utao0  = parameters [11]
    utao1  = parameters [12]
    # 偏导函数
    dux1_dx = uc*(ue*( uk0*ux3**2 + uk1*ux3**4 ))*math.cos(ue*ux1*( uk0*ux3**2 + uk1*ux3**4 ))
    dux1_dy = ua
    dux1_dz = uc*(ue*ux1*( 2*uk0*ux3 + 4*uk1*ux3**3 ))*math.cos(ue*ux1*( uk0*ux3**2 + uk1*ux3**4 ))
    dux1_du = 0
    dux2_dx = ub
    dux2_dy = ud*math.cos(ux4)
    dux2_dz = 0
    dux2_du = -ud*ux2*math.sin(ux4)
    dux3_dx = 0
    dux3_dy = utao0
    dux3_dz = 1
    dux3_du = 0
    dux4_dx = 0
    dux4_dy = utao1
    dux4_dz = 0
    dux4_du = 1
    # 雅可比矩阵
    Jac = np.array([
        [dux1_dx,dux1_dy,dux1_dz,dux1_du],
        [dux2_dx,dux2_dy,dux2_dz,dux2_du],
        [dux3_dx,dux3_dy,dux3_dz,dux3_du],
        [dux4_dx,dux4_dy,dux4_dz,dux4_du]
    ])
    return Jac

def mod_5D_chaotic_func( parameters ):
    # 根据不同的模型输入不同的数据
    ux1    = parameters [0]
    ux2    = parameters [1]
    ux3    = parameters [2]
    ux4    = parameters [3]
    ux5    = parameters [4]
    # 模型已知的参数
    ua     = parameters [5]
    ub     = parameters [6]
    uc     = parameters [7]
    ud     = parameters [8]
    ue     = parameters [9]
    uf     = parameters [10]
    uk0    = parameters [11]
    uk1    = parameters [12]
    utao0  = parameters [13]
    utao1  = parameters [14]
    utao2  = parameters [15]

    # 混沌系统
    ux1n = ua*ux2 + uc * math.sin( ue*ux1*( uk0*ux3**2 + uk1*ux3**4 ))
    ux2n = ub*ux1 + ud*ux2 * math.sin(ux4) + uf*ux2*math.cos(ux5)
    # ux2n = ub*ux1 + ud * math.cos( ux2*( uk0*ux4**2 + uk1*ux4**4 ) )
    ux3n = ux3 + utao0 * ux1
    # ux3n = ux3 + utao0 * ux2
    ux4n = ux4 + utao1 * ux2
    ux5n = ux5 + utao2 * ux2

    # 返回迭代值
    parameters_n = np.array(( ux1n, ux2n, ux3n, ux4n, ux5n, ua, ub, uc, ud, ue, uf, uk0, uk1, utao0, utao1, utao2 ))
    return parameters_n

def mod_5D_chaotic_func_Jac ( parameters ):
    # 根据不同的模型输入不同的数据
    ux1    = parameters [0]
    ux2    = parameters [1]
    ux3    = parameters [2]
    ux4    = parameters [3]
    ux5    = parameters [4]
    # 模型已知的参数
    ua     = parameters [5]
    ub     = parameters [6]
    uc     = parameters [7]
    ud     = parameters [8]
    ue     = parameters [9]
    uf     = parameters [10]
    uk0    = parameters [11]
    uk1    = parameters [12]
    utao0  = parameters [13]
    utao1  = parameters [14]
    utao2  = parameters [15]
    # 偏导函数
    dux1_dx = uc*ue*( uk0*ux3**2 + uk1*ux3**4 )*math.cos(ue*ux1*( uk0*ux3**2 + uk1*ux3**4 ))
    dux1_dy = ua
    dux1_dz = uc*ue*ux1*( 2*uk0*ux3 + 4*uk1*ux3**3 ) * math.cos(ue*ux1*( uk0*ux3**2 + uk1*ux3**4 ))
    dux1_du = 0
    dux1_dv = 0

    dux2_dx = ub
    dux2_dy = ud*math.sin(ux4)+uf*math.cos(ux5)
    dux2_dz = 0
    dux2_du = ud*ux2*math.cos(ux4)
    dux2_dv = -uf*ux2*math.sin(ux5)

    dux3_dx = utao0
    dux3_dy = 0
    dux3_dz = 1
    dux3_du = 0
    dux3_dv = 0

    dux4_dx = 0
    dux4_dy = utao1
    dux4_dz = 0
    dux4_du = 1
    dux4_dv = 0

    dux5_dx = 0
    dux5_dy = utao2
    dux5_dz = 0
    dux5_du = 0
    dux5_dv = 1

    # 雅可比矩阵
    Jac = np.array([
        [dux1_dx,dux1_dy,dux1_dz,dux1_du,dux1_dv],
        [dux2_dx,dux2_dy,dux2_dz,dux2_du,dux2_dv],
        [dux3_dx,dux3_dy,dux3_dz,dux3_du,dux3_dv],
        [dux4_dx,dux4_dy,dux4_dz,dux4_du,dux4_dv],
        [dux5_dx, dux5_dy, dux5_dz, dux5_du, dux5_dv]
    ])
    return Jac

def sin_map_new(parameters):
    x = parameters[0]
    mu = parameters[1]
    return np.array( ( mu * np.sin(np.pi * x) , mu))

def sin_map_deriv_new(parameters):
    x = parameters[0]
    mu = parameters[1]
    return mu * np.cos( np.pi * x) * np.pi

def mod_4d_TBMHM( parameters ):
    # 根据不同的模型输入不同的数据
    ux1    = parameters [0]
    ux2    = parameters [1]
    ux3    = parameters [2]
    ux4    = parameters [3]
    # 模型已知的参数
    ua     = parameters [4]
    ub     = parameters [5]
    uc     = parameters [6]
    ud     = parameters [7]
    ue     = parameters [8]
    uk0    = parameters [9]
    uk1    = parameters [10]
    uk2    = parameters [11]
    utao   = parameters [12]

    # 混沌系统
    ux1n = ua*ux2 + uc * math.sin( ue * ux2 * ( uk0 + uk1 * ux3 + uk2 * ux3 * ux3 ) )
    ux2n = ub * ux1 + ud * ux2 * math.cos( ux4 )
    ux3n = ux3 + utao * ux2
    ux4n = ux4 + utao * ux2

    # 返回迭代值
    parameters_n = np.array(( ux1n, ux2n, ux3n, ux4n, ua, ub, uc, ud, ue, uk0, uk1, uk2 , utao ))
    return parameters_n

def mod_4d_TBMHM_Jac( parameters ):
    # 根据不同的模型输入不同的数据
    ux1    = parameters [0]
    ux2    = parameters [1]
    ux3    = parameters [2]
    ux4    = parameters [3]
    # 模型已知的参数
    ua     = parameters [4]
    ub     = parameters [5]
    uc     = parameters [6]
    ud     = parameters [7]
    ue     = parameters [8]
    uk0    = parameters [9]
    uk1    = parameters [10]
    uk2    = parameters [11]
    utao   = parameters [12]
    # 偏导函数
    dux1_dx = 0
    dux1_dy = ua +  uc * ue  * ( uk0 + uk1 * ux3 + uk2 * ux3 ** 2 ) * math.cos( ue * ux2 * ( uk0 + uk1 * ux3 + uk2 * ux3 * ux3 ))
    dux1_dz = uc *ue * ux2 * ( uk1 + 2 * uk2 * ux3 ) * math.cos( ue * ux2 * ( uk0 + uk1 * ux3 + uk2 * ux3 * ux3  ))
    dux1_du = 0
    dux2_dx = ub
    dux2_dy = ud*math.cos(ux4)
    dux2_dz = 0
    dux2_du = - ud * ux2 * math.sin(ux4)
    dux3_dx = 0
    dux3_dy = utao
    dux3_dz = 1
    dux3_du = 0
    dux4_dx = 0
    dux4_dy = utao
    dux4_dz = 0
    dux4_du = 1
    # 雅可比矩阵
    Jac = np.array([
        [dux1_dx,dux1_dy,dux1_dz,dux1_du],
        [dux2_dx,dux2_dy,dux2_dz,dux2_du],
        [dux3_dx,dux3_dy,dux3_dz,dux3_du],
        [dux4_dx,dux4_dy,dux4_dz,dux4_du]
    ])
    return Jac

def Logistic_map_new(parameters):
    x = parameters[0]
    mu = parameters[1]
    return np.array (( mu * x * (1 - x) , mu))


def Logistic_map_deriv_new(parameters):
    x = parameters[0]
    mu = parameters[1]
    return mu*(1-2*x)
