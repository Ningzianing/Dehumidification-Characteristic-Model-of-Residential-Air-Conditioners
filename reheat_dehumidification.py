import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import CoolProp.CoolProp as HAP
import sympy as sp
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from scipy import interpolate
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar

def T_evp_cal_JIS(Troom,RHroom,L_total,BP,Q_in,): #蒸発温度計算
    dew_point_temperature  = HAP.HAPropsSI('Tdp','T', Troom+273.15,'P', 101325,'R', RHroom / 100.0)-273.15
    M_in = Q_in / 3600 * 353.25/(Troom+273.15)
    M_evp = (1 - BP) * M_in
    T_evp = Troom - L_total/(M_evp * 1005)
    if T_evp < dew_point_temperature:
        h_evp = HAP.HAPropsSI('H','T', Troom+273.15,'P', 101325,'R', RHroom / 100.0) - L_total / M_evp
        T_evp = HAP.HAPropsSI('T','H', h_evp,'P', 101325,'R', 1) -273.15
    return T_evp+273.15

def T_cnd_cal_JIS(Tout,L,BP,Q_out,P):# 凝縮温度計算
    M_out = Q_out / 3600 * 353.25/(Tout+273.15)
    M_cnd = (1 - BP) * M_out
    T_cnd = Tout + (L + P)/(M_cnd * 1005)
    return T_cnd+273.15

def cal_Pc(BP_evp,BP_cnd,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,):
    Tout,Troom,RHroom = 35,27,47#JIS条件の室内外条件
    P_c = sp.symbols('P_c')
    R_low = ((L_low / P_low) * L_low / (L_low - (L_low / P_low) * P_c)) / (
                T_evp_cal_JIS(Troom, RHroom, L_low, BP_evp, Q_in) / (
                    T_cnd_cal_JIS(Tout, L_low, BP_cnd, Q_out, P_low) - T_evp_cal_JIS(Troom, RHroom, L_low, BP_evp, Q_in)))
    R_medium = ((L_medium / P_medium) * L_medium / (L_medium - (L_medium / P_medium) * P_c)) / (
                T_evp_cal_JIS(Troom, RHroom, L_medium, BP_evp, Q_in) / (
                    T_cnd_cal_JIS(Tout, L_medium, BP_cnd, Q_out, P_medium) - T_evp_cal_JIS(Troom, RHroom, L_medium, BP_evp, Q_in)))
    solution = sp.solve(R_low - R_medium, P_c)
    P_c = solution[-1]
    return P_c
################为了算P_c#################

def quadratic_function(x, a, b, c):
    return a * x ** 2 + b * x + c

def Min_Qtotal_function(x, M_in_a, M_in_b):
    return M_in_a * x + M_in_b

def delta_t_P_func(P_total, delta_t_a, delta_t_b):
    val = delta_t_a * P_total + delta_t_b
    return max(0, min(val, 5))

###################################
# get dew point after optimization
###################################
def get_w_from_tk(t_K, P):
    # Your Goff-Gratch formula
    Ps = 10 ** (
        -7.90298 * (373.16 / t_K - 1)
        + 5.02808 * np.log10(373.16 / t_K)
        - 1.3816e-7 * (10 ** (11.344 * (1 - t_K / 373.16)) - 1)
        + 8.1328e-3 * (10 ** (-3.49149 * (373.16 / t_K - 1)) - 1)
        + np.log10(1013.246)
    ) * 100 
    return 0.622 * Ps / (P - Ps)
def get_dew_point(target_w, P=101325):
    # # Objective function: we want this to be 0
    # func = lambda t: get_w_from_tk(t, P) - target_w
    # # Adjust range based on your expected environmental conditions
    # t_solution = brentq(func, 253.15, 323.15) # 这里容易报错的
    # 改成找最小值问题
    def objective_min(t, P, target_w):
        try:
            error = get_w_from_tk(t, P) - target_w
            return error**2 
        except:
            # 如果物性计算报错（比如温度超出物性库范围），返回一个极大的惩罚值
            return 1e18
    # 2. 在指定区间内寻找最小值
    res = minimize_scalar(objective_min, bounds=(253.15, 323.15), args=(P, target_w), method='bounded')
    if res.success:
        t_solution = res.x
        residual = res.fun 
    else:
        print("get_dew_point没有成功最小化")
        t_solution = np.nan
    return t_solution-273.15 # return ℃

def solve_Mout(T_out, X_xiru, T_xiru, Q_sensible, Q_latent, M_in, P_total, gamma_cooled, gamma_reheat, BP, delta_t_a, delta_t_b): ## Mout,Tevp,Tcndinの計算
    # 这里的M_in用实测数据代替就行
    delta_t = delta_t_P_func(P_total, delta_t_a, delta_t_b) # 室外机冷凝温度-室内机冷凝温度
    X_evp = X_xiru - Q_latent/(M_in*gamma_cooled*(1-BP)*2501000)
    try:
        T_evp = HAP.HAPropsSI('D', 'W', X_evp, 'P', 101325, 'B', 293.15) - 273.15 # 这里容易报错的
    except:
        T_evp = get_dew_point(float(X_evp)) # 这里容易报错的

    T_cndin = (M_in*gamma_cooled*(1-BP)*1005*(T_xiru - T_evp) + M_in*(1-gamma_cooled)*(1-BP)*1005*T_xiru + M_in*gamma_cooled*gamma_reheat*(1-BP)*1005*T_evp - Q_sensible)/(M_in*(1-gamma_cooled)*(1-BP)*1005 + M_in*gamma_cooled*gamma_reheat*(1-BP)*1005)
    M_out = (Q_sensible + Q_latent + P_total)/(1005*(T_cndin + delta_t -T_out)*(1-BP))
    return X_evp, T_evp, T_cndin, M_out

def R_Q_relation(df, T_evp_cals, T_cndin_cals, gamma_cooled, gamma_reheat, BP, P_c):
    # 最小二乘拟合R_dehumid-Q_real关系，只和
    Q_cnd_cals = []
    Q_real_cals = []
    carnoefficients = []
    realefficients = []
    R_dehumid_cals = []

    for y in range(len(df)):
        
        Q_cnd_cal = df['M_in'][y] * (1-gamma_cooled) * (1 - BP) * 1005 * (T_cndin_cals[y]-df["Txiru"][y]) + df['M_in'][y]*gamma_cooled*gamma_reheat*(1-BP)*1005*(T_cndin_cals[y] - T_evp_cals[y])
        Q_cnd_cals.append(Q_cnd_cal)

        Q_real_cal = df["Q_sensible"][y] + df["Q_latent"][y] + Q_cnd_cal # 压缩机实际处理的热量
        Q_real_cals.append(Q_real_cal)

        carnoefficient = (T_evp_cals[y]+273.15)/(T_cndin_cals[y]-T_evp_cals[y])
        carnoefficients.append(carnoefficient)

        realefficient = Q_real_cal/(df["Totalenergy"][y]-P_c) 
        realefficients.append(realefficient)

        R_dehumid_cal = realefficient/carnoefficient
        R_dehumid_cals.append(R_dehumid_cal)

    Q_real_cals_np = np.array(Q_real_cals,dtype = float)
    R_dehumid_cals_np = np.array(R_dehumid_cals, dtype = float)
    coeffs = np.polyfit(Q_real_cals_np, R_dehumid_cals_np, deg=2)

    print(coeffs)

    poly_func = np.poly1d(coeffs)

    return poly_func


def iterate_T_evp(T_evp, M_in, gamma_cooled, gamma_reheat, BP, X_xiru, T_xiru, T_cndin, Q_total, P=101325):
    T_evp_K = T_evp + 273.15
    if T_evp_K <= 0:
        return -1e10 
    
    # Goff-Gratch
    inner_val = (
        -7.90298 * (373.16 / T_evp_K - 1)
        + 5.02808 * np.log10(373.16 / T_evp_K)
        - 1.3816e-7 * (10 ** (11.344 * (1 - T_evp_K / 373.16)) - 1)
        + 8.1328e-3 * (10 ** (-3.49149 * (373.16 / T_evp_K - 1)) - 1)
        + np.log10(1013.246)
    )
    
    Ps = 10 ** inner_val * 100 
    Ps = min(Ps, P * 0.99)
    X_evp = 0.622 * Ps / (P - Ps)
    
    Q_latent_cal = M_in * gamma_cooled * (1 - BP) * 2501000 * (X_xiru - X_evp)
    Q_sensible_cal = Q_total-Q_latent_cal
    T_evp_cal = (Q_sensible_cal - M_in * gamma_cooled * (1 - BP) * 1005 * T_xiru + M_in * (1 - gamma_cooled) * (1 - BP) * 1005 * (T_cndin - T_xiru) + M_in * gamma_cooled * gamma_reheat * (1-BP) * 1005 * T_cndin) / (M_in * gamma_cooled * gamma_reheat * (1-BP) * 1005 - M_in * gamma_cooled * (1-BP) * 1005)
    return float(T_evp_cal)

def solve_main_eqs(M_in, X_xiru, T_xiru, T_out, Q_total, P_aircon, gamma_cooled, gamma_reheat, BP, a, b, delta_t_a, delta_t_b, 
                        Q_latent_low=500, Q_latent_high=1200,
                        T_evp_tolerance=0.5, max_iter=20): # solve the equation to find X_evp, T_evp, T_cndin, Q_latent
    M_out = a * (Q_total + P_aircon) + b
    delta_t = delta_t_P_func(P_aircon, delta_t_a, delta_t_b)
    T_cndin = (Q_total + P_aircon)/((1-BP)*M_out*1005) + T_out - delta_t

    # ######################
    # # 换一种找T_evp的方法，迭代着解
    # ######################
    continue_T_evp = True
    T_evp_solution = 10
    T_evp_iterate= 0
    T_evp_max = 20
    T_evp_min = 0
    while continue_T_evp == True and T_evp_iterate <= max_iter:
        T_evp_solution_cal = iterate_T_evp(T_evp_solution, M_in, gamma_cooled, gamma_reheat, BP, X_xiru, T_xiru, T_cndin, Q_total)
        T_evp_solution_cal = max(min(T_evp_solution_cal, T_evp_max), T_evp_min)
        if abs(T_evp_solution - T_evp_solution_cal) <= T_evp_tolerance:
            T_evp_solution = (T_evp_solution + T_evp_solution_cal) / 2
            continue_T_evp = False       
        else:
            T_evp_solution = (T_evp_solution + T_evp_solution_cal) / 2
            T_evp_iterate += 1    
    if T_evp_iterate==max_iter:
        print("P_aircon:", P_aircon, "Q_total:", Q_total, "gamma_cooled, gamma_reheat, BP:", gamma_cooled, gamma_reheat, BP, "当前solve_main_eqs未收敛" )

    T_evp_solution_K = T_evp_solution + 273.15
    Ps_solution = 10 ** (
        -7.90298 * (373.16 / T_evp_solution_K - 1)
        + 5.02808 * np.log10(373.16 / T_evp_solution_K)
        - 1.3816e-7 * (10 ** (11.344 * (1 - T_evp_solution_K / 373.16)) - 1)
        + 8.1328e-3 * (10 ** (-3.49149 * (373.16 / T_evp_solution_K - 1)) - 1)
        + np.log10(1013.246)
    ) * 100 

    X_evp_solution = 0.622 * Ps_solution / (101325 - Ps_solution)
    
    Q_latent_solution = M_in*gamma_cooled*(1-BP)*2501000*(X_xiru-X_evp_solution)
    return X_evp_solution, T_evp_solution, T_cndin, Q_latent_solution


def solve_P(P_initial, X_xiru, T_xiru, T_out, Q_total, P_tolerance, Min_a, Min_b, gamma_cooled, gamma_reheat, BP, a, b, poly_func, P_c, delta_t_a, delta_t_b,max_iterate = 20): #消費電力の計算
        M_in = Min_Qtotal_function(Q_total, Min_a, Min_b)

        continuecal = True
        P = P_initial
        iterate = 0

        # 设置上下限
        P_min = 100 
        P_max = 400  

        while continuecal == True:
            # print(P)
            X_evp, T_evp, T_cndin, Q_latent = solve_main_eqs(M_in, X_xiru, T_xiru, T_out, Q_total, P, gamma_cooled, gamma_reheat, BP, a, b, delta_t_a, delta_t_b, Q_latent_low=0, Q_latent_high=1500) # 也可以by grid
            Q_cndin = M_in*(1-gamma_cooled)*(1-BP)*1005*(T_cndin - T_xiru) + M_in*gamma_cooled*gamma_reheat*(1-BP)*1005*(T_cndin-T_evp)

            Q_total_compressor = Q_total + Q_cndin
            R_dehumid = poly_func(Q_total_compressor)
            P_total = (Q_total_compressor / ((T_evp+273.15)*R_dehumid/(T_cndin-T_evp))) + P_c
            P_total = max(min(P_total, P_max), P_min)

            if abs(P-P_total) < P_tolerance or iterate > max_iterate:
                continuecal = False
                break
            else:
                P_new = (P + P_total)/2
                P = max(min(P_new, P_max), P_min)
                iterate += 1
            if iterate >= max_iterate:
                print("Q_total:", Q_total, "gamma_cooled, gamma_reheat, BP:", gamma_cooled, gamma_reheat, BP, "当前solve_P未收敛")

        return max(min((P + P_total)/2, P_max), P_min), T_evp, T_cndin, Q_latent, (Q_total - Q_latent) / Q_total 

"""
can only be used in daikin ax
"""
class reheat_aircon():
    def __init__(self, BP_evp,BP_cnd,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,oneday_testdata, construct_model = True):#室内機bypass　factor （电中研） , 室外機bypass factor （电中研）, 最小能力、定格能力、最大能力,最小EC,定格EC,最大EC,室内機風量（仕様書）、室外機風量（仕様書）,一日の実験データ

        df = pd.read_csv(oneday_testdata)
        print(df.head(2))
        
        # 实际测试中，如果这三列中有任何一列元素是0，就把这行删掉
        df = df[~((df['Q_sensible'] == 0) | (df['Q_latent'] == 0) | (df['Q_total'] == 0))].reset_index(drop=True)
        
        # 【--------------------------建立M_in和Q_total的关系-----------------------------】
        x_data = df['Q_total'].values
        y_data = df['M_in'].values
        popt, pcov = curve_fit(Min_Qtotal_function, x_data, y_data, p0=[1.0, 0.0])
        self.Min_a = popt[0] # 斜率
        self.Min_b = popt[1] # 截距

        x_fit = np.linspace(x_data.min(), x_data.max(), 100)
        # 使用你拟合出的 self.d (即底数 a) 计算对应的 y
        y_fit = Min_Qtotal_function(x_fit, self.Min_a, self.Min_b)
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, color='blue', label='Original Data', alpha=0.6)
        plt.plot(x_fit, y_fit, color='red', linewidth=2)
        plt.title('Q_total vs M_in Fitting')
        plt.xlabel('Q_total')
        plt.ylabel('M_in')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        # 【--------------------------贝叶斯找gamma_cooled, gamma_reheat, BP-----------------------------】
        space = [
            Real(0, 0.5, name='gamma_cooled'),
            Real(0.1, 0.5, name='BP'),
            Real(0, 0.5, name='gamma_reheat'), 
            Real(0, 5/300, name='delta_t_a'),
            Real(0, 3, name='delta_t_b')
        ]

        # 定义目标函数
        @use_named_args(space)
        def objective(**params):
            mse_all = []
            for i in range(20):
                df_shuffled = df.sample(frac=1).reset_index(drop=True)
                n = len(df_shuffled)
                split_index = n // 2
                df_first_half = df_shuffled.iloc[:split_index].copy().reset_index(drop=True)
                df_second_half = df_shuffled.iloc[split_index:].copy().reset_index(drop=True)

                gamma_cooled = params['gamma_cooled']
                BP = params['BP']
                gamma_reheat = params['gamma_reheat']
                delta_t_a = params['delta_t_a']
                delta_t_b = params['delta_t_b']

                # P_c跟着BP走
                P_c = cal_Pc(BP,BP,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out)

                M_out_solutions = []
                T_evp_solutions = []
                T_cndin_solutions = []
                for k in range(len(df_first_half)):
                    try:
                        X_evp, T_evp, T_cndin, M_out = solve_Mout(
                            df_first_half['Tout'][k], 
                            df_first_half['Xxiru'][k], 
                            df_first_half['Txiru'][k], 
                            df_first_half['Q_sensible'][k], 
                            df_first_half['Q_latent'][k], 
                            df_first_half['M_in'][k], 
                            df_first_half['Totalenergy'][k], 
                            gamma_cooled, gamma_reheat, BP, delta_t_a, delta_t_b
                        )
                        # 检查物理合理性，防止 inf 或 nan 混入拟合
                        if np.isinf(M_out) or np.isnan(M_out):
                            return 1e9
                            
                        M_out_solutions.append(M_out)
                        T_evp_solutions.append(T_evp)
                        T_cndin_solutions.append(T_cndin)
                    except:
                        print("solve_Mout报错")
                        return 1e9 # 只要有一行算崩了，这组参数就不可用

                # if any(m < 0 for m in M_out_solutions):
                #     return 1e6

                Load_out = (df_first_half["Q_sensible"] + df_first_half["Q_latent"] + df_first_half["Totalenergy"]).tolist()

                try:
                    coeffs = np.polyfit(Load_out, M_out_solutions, 1)  
                    a, b = coeffs
                    print("M_out a:", a, "M_out b:", b)

                except:
                    return 1e6
                
                try:
                    poly_func = R_Q_relation(df_first_half, T_evp_solutions, T_cndin_solutions, gamma_cooled, gamma_reheat, BP, P_c)
                except:
                    return 1e6

                P_totals = []
                Q_latents = []

                # 让训练集+验证集上的消费电力mse最小
                # 万一贝叶斯到一半报错了就狠狠地罚
                PUNISHMENT_VALUE = 1e9 

                for e in range(len(df_second_half)):
                    try:
                        # 尝试运行物理求解器
                        P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = solve_P(
                            200, 
                            df_second_half["Xxiru"].iloc[e], 
                            df_second_half["Txiru"].iloc[e], 
                            df_second_half["Tout"].iloc[e], 
                            df_second_half["Q_total"].iloc[e], 
                            15, self.Min_a, self.Min_b, gamma_cooled, gamma_reheat, BP, a, b, poly_func, P_c, delta_t_a, delta_t_b
                        )
                        if Q_latent_cal < 0 or Q_latent_cal > df_second_half["Q_total"].iloc[e] or SHF < 0 or T_evp_apply > T_cnd_apply:
                            print("出现不符合物理的数值：", "Q_latent_cal, T_evp_apply, T_cnd_apply:", Q_latent_cal, T_evp_apply, T_cnd_apply)
                            P_totals.append(PUNISHMENT_VALUE)
                            Q_latents.append(PUNISHMENT_VALUE)
                        else:
                            P_totals.append(P)
                            Q_latents.append(Q_latent_cal)
                        
                    except Exception as err:
                        # 万一报错（比如之前的 brentq 符号错误），执行惩罚逻辑
                        print(f"Warning: Row {e} failed to solve. Applying punishment. Error: {err}")
                        
                        # 给 P 赋一个离谱的值，这样算出来的 MSE 会变得巨大
                        P_totals.append(PUNISHMENT_VALUE)
                        Q_latents.append(PUNISHMENT_VALUE) # 潜热可以设为 0 或其他占位符
                        continue

                # 计算 MSE
                mse = mean_squared_error(P_totals, df_second_half['Totalenergy'].tolist()) + 0.05 * mean_squared_error(Q_latents, df_second_half['Q_latent'].tolist())
                mse_all.append(mse)
                i += 1
            return float(sum(mse_all) / len(mse_all))
        
        if construct_model == True:
            # Optimization 
            result = gp_minimize(
                objective,
                space,
                n_calls=100,              # 总评估次数
                n_random_starts=5,      # 随机初始点数量
                n_jobs=-1,              # 使用所有CPU核心
                noise=1e-10,            # 假设噪声水平
                acq_func='EI',          # 采集函数：Expected Improvement
                verbose=True
            )

            # 打印结果
            print("Best parameters：")
            print(f"gamma_cooled: {result.x[0]}")
            print(f"BP: {result.x[1]}")
            print(f"gamma_reheat: {result.x[2]}")
            print(f"delta_t_a: {result.x[3]}")
            print(f"delta_t_b: {result.x[4]}")
            print(f"min MSE: {result.fun}")

            self.gamma_cooled = result.x[0]
            self.BP = result.x[1]
            self.gamma_reheat = result.x[2]
            self.delta_t_a = result.x[3]
            self.delta_t_b = result.x[4]
        
        else:
            # self.gamma_cooled = 0.4799914500614519
            # self.BP = 0.1
            # self.gamma_reheat = 0.5
            # self.delta_t_a = 0.004879627118704553
            # self.delta_t_b = 3.0

            self.gamma_cooled = 0.43874113146466137
            self.BP = 0.1
            self.gamma_reheat = 0.5
            self.delta_t_a = 0.0
            self.delta_t_b = 3.0

        # 【---------------------------Moutを計算し，Moutと室外機処理熱量の关系を求める------------------------------】
        M_out_solutions_best = []
        T_evp_solutions_best = []
        T_cndin_solutions_best = []
        for k in range(len(df)):
            X_evp, T_evp, T_cndin, M_out = solve_Mout(df['Tout'][k], df['Xxiru'][k], df['Txiru'][k], df['Q_sensible'][k], df['Q_latent'][k], df['M_in'][k], df['Totalenergy'][k], self.gamma_cooled, self.gamma_reheat, self.BP, self.delta_t_a, self.delta_t_b)
            M_out_solutions_best.append(M_out)
            T_evp_solutions_best.append(T_evp)
            T_cndin_solutions_best.append(T_cndin)

        Load_out_best = (df["Q_sensible"] + df["Q_latent"] + df["Totalenergy"]).to_list() # 室外机真正处理的热量
        # 最小二乘拟合：多项式1次，等价于线性拟合 y = a * x + b
        coeffs = np.polyfit(Load_out_best, M_out_solutions_best, 1)  
        print(coeffs)
        self.a, self.b = coeffs

        self.P_c = cal_Pc(self.BP,self.BP,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out)
        print("P_c:,", self.P_c)

        # # 看一下拟合效果
        M_out_fit = []
        for i in range(len(df)):
            M_out_fit.append(self.a*Load_out_best[i]+self.b)
        plt.figure(figsize=(10,10)) 
        plt.scatter((df['Q_sensible'] + df["Q_latent"] + df["Totalenergy"]).to_list(), M_out_solutions_best, color='blue')
        plt.title('Mout-Load_out relation', fontsize=16)
        plt.plot(Load_out_best,M_out_fit, color='red')
        plt.xlabel('Load_out', fontsize=20)
        plt.ylabel('Mout', fontsize=20)  
        plt.grid(True)
        plt.show() 

        # 【---------------------------R_dehumid-Q_realの关系を求める------------------------------】
        self.poly_func = R_Q_relation(df, T_evp_solutions_best, T_cndin_solutions_best, self.gamma_cooled, self.gamma_reheat, self.BP, self.P_c)
    
    # 【------------------------------现在空调本身的性质都找到了，应用一下：---------------------------------】
    def output(self, P_initial, X_xiru, T_xiru, T_out, Q_total, P_tolerance, max_iterate = 10): #求电力，把能输出的都输出，用迭代
        P_cal, T_evp, T_cndin, Q_latent, SHF = solve_P(P_initial, X_xiru, T_xiru, T_out, Q_total, P_tolerance, self.Min_a, self.Min_b,  self.gamma_cooled, self.gamma_reheat, self.BP, self.a, self.b, self.poly_func, self.P_c, self.delta_t_a, self.delta_t_b, max_iterate)
        return P_cal, T_evp, T_cndin, Q_latent, SHF


if __name__ == "__main__":
    experiment_data = r"data\0615data_processed_0_7.csv"
    ac = reheat_aircon(0.1,0.25,500,2200,3300,115,425,960,906,2160,experiment_data) #室内機bypass　factor （电中研） , 室外機bypass factor （电中研）, 最小能力、定格能力、最大能力,最小EC,定格EC,最大EC,室内機風量（仕様書）、室外機風量（仕様書）,一日の実験データ

    P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = ac.output(250,0.010613295862459, 24.2, 25.1, 759, 10, 30)
    print(P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF)


    test_data_XYplot = r"data\0614_0810_data_processed_0_7.csv"
    df_test1 = pd.read_csv(test_data_XYplot)
    df_test1['Time'] = pd.to_datetime(df_test1['Time'])
    start_time = "2025-08-10 00:00:00"
    end_time = "2025-08-10 23:50:00"
    mask = (df_test1['Time'] >= start_time) & (df_test1['Time'] <= end_time)
    df_test1 = df_test1.loc[mask].reset_index(drop=True)

    df_test = pd.concat([df_test1.iloc[0:18], df_test1.iloc[126:]])
    df_test = df_test.reset_index(drop=True)
    print(len(df_test))

    P_totals = []
    Q_latents = []
    T_evps = []
    T_cndins = []
    SHFs = []

    for e in range(len(df_test)):
        print("--------------------------------------------------------------")
        print("real energy",df_test["Totalenergy"][e],e)
        if df_test["Totalenergy"][e] < 10:
            P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = 0,0,0,0,0
        else:
            P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = ac.output(300, df_test["Xxiru"][e], df_test["Txiru"][e], df_test["Tout"][e], df_test["Q_sensible"][e] + df_test["Q_latent"][e], 10)
        print("Calculated energy:", P)
        P_totals.append(P)
        Q_latents.append(Q_latent_cal)
        T_evps.append(T_evp_apply)
        T_cndins.append(T_cnd_apply)
        SHFs.append(SHF)

    # 能耗结果
    import matplotlib.dates as mdates
    plt.figure(figsize=(12, 6))
    plt.plot(df_test['Totalenergy'], label='Measured energy', color='blue')
    plt.plot(P_totals, label='Calculated energy', color='orange')
    # plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.ylabel('Energy (W)', fontsize=14)
    plt.title('Measured vs Calculated Energy Over Time', fontsize=16)
    plt.ylim(0,300)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    y_true = df_test['Totalenergy'].values
    y_pred = np.array(P_totals)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse:.4f}")

    # 潜热结果
    plt.figure(figsize=(12, 6))
    plt.plot(df_test['Q_latent'], label='Measured Q_latent', color='blue')
    plt.plot(Q_latents, label='Calculated Q_latent', color='orange')
    # plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.ylabel('Qlatent (W)', fontsize=14)
    plt.title('Measured vs Calculated Qlatent Over Time', fontsize=16)
    plt.ylim(0,1000)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    


    
    












