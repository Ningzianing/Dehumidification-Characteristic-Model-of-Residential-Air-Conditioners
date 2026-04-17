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

# 这里对P_c做文章。。。。

def T_evp_cal_sensible(Troom,L_sensible,BP_in,Q_in,): ##Q_in是室内机是吸入风量，单位是m3/h
    M_in = Q_in / 3600 * 353.25/(Troom+273.15)
    M_evp = (1 - BP_in) * M_in
    T_evp = Troom - L_sensible/(M_evp * 1005)
    return T_evp+273.15 ##这是室内机的蒸发温度,单位是K

def T_cnd_cal_sensible(Tout,L_total,BP_out,Q_out,P): # Q_out是室外机吸入风量
    M_out = Q_out / 3600 * (353.25/(Tout+273.15))
    M_cnd = (1 - BP_out) * M_out
    T_cnd = Tout + (L_total + P)/(M_cnd * 1005)
    return T_cnd + 273.15  ##这是室外机的冷媒凝缩温度，单位是K

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

def cal_R_Q_relation(BP_in,BP_out,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,P_c):
    Tout,Troom,RHroom = 35,27,47#JIS条件の室内外条件
    R_low    = ((L_low/P_low) *L_low /(L_low-(L_low/P_low)* P_c))  / (T_evp_cal_JIS(Troom,RHroom, L_low,BP_in,Q_in)/(T_cnd_cal_JIS(Tout,L_low,BP_out,Q_out,P_low) - T_evp_cal_JIS(Troom,RHroom,L_low,BP_in,Q_in)))
    R_medium = ((L_medium/P_medium)*L_medium/(L_medium-(L_medium/P_medium)* P_c)) / (T_evp_cal_JIS(Troom,RHroom,L_medium,BP_in,Q_in)/(T_cnd_cal_JIS(Tout,L_medium,BP_out,Q_out,P_medium)- T_evp_cal_JIS(Troom,RHroom,L_medium,BP_in,Q_in)))
    R_high   = ((L_high/P_high)    *L_high  /(L_high-(L_high/P_high)* P_c)) / (T_evp_cal_JIS(Troom,RHroom,L_high,BP_in,Q_in)/(T_cnd_cal_JIS(Tout,L_high,BP_out,Q_out,P_high) - T_evp_cal_JIS(Troom,RHroom,L_high,BP_in,Q_in)))
    # 输入三个点的坐标
    x_data = [L_low,L_medium,L_high]
    y_data = [R_low,R_medium,R_high]
    # 进行二次拟合
    params, covariance = curve_fit(quadratic_function, x_data, y_data)
    # 得到二次拟合的系数
    a, b, c = params
    return a,b,c

def quadratic_function(x, a, b, c):
    return a * x ** 2 + b * x + c

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
    return t_solution # return K

def L_sensible_latent_cal(L_total,Troom,Xroom,BP_in,Q_in, tolerance=10, max_iterations=20):
    M_in = Q_in / 3600 * 353.25/(Troom+273.15)
    T_evp = T_evp_cal_sensible(Troom,L_total,BP_in,Q_in) # K

    T_dew_room = get_dew_point(float(Xroom)) # K

    if T_evp >= T_dew_room:
        Lsensible = L_total
        T_evp_final = T_evp #K
    else:
        iteration = 0
        Lsensible = 0.8*L_total 

        while iteration < max_iterations:
            T_evp = T_evp_cal_sensible(Troom,Lsensible,BP_in,Q_in)
            X_evp = HAP.HAPropsSI('W', 'T', T_evp, 'P', 101325, 'R', 1.0)
            Llatent = M_in*(1-BP_in)*2501000*(Xroom-X_evp)
            Lsensible_cal = L_total - Llatent
            # print("Lsensible:",Lsensible,"L_sensible_cal:",Lsensible_cal)

            # print("Lsensible:",Lsensible, "Lsensible_cal:",Lsensible_cal)

            if abs(Lsensible - Lsensible_cal) < tolerance:
                Lsensible = 0.5*(Lsensible + Lsensible_cal)
                T_evp_final = T_evp_cal_sensible(Troom,Lsensible,BP_in,Q_in) #K
                break

            Lsensible = 0.5*(Lsensible + Lsensible_cal)
            T_evp_final = T_evp_cal_sensible(Troom,Lsensible,BP_in,Q_in)
            iteration += 1

    # print("Lsensible:",Lsensible, "T_evp:", T_evp_final)
    return Lsensible, T_evp_final #K

def find_P_total(BP_in, BP_out, a,b,c, P_c, Q_out, L_total,Troom,Xroom,Tout, Q_in_apply):
    M_in = Q_in_apply / 3600 * 353.25/(Troom+273.15)
    M_out = Q_out / 3600 * 353.25/(Tout+273.15)

    Lsensible_apply, T_evp_apply = L_sensible_latent_cal(L_total,Troom,Xroom,BP_in, Q_in_apply, tolerance=5, max_iterations=20)
    R = quadratic_function(L_total, a, b, c)

    A = (L_total**2/(M_out*1005*(1-BP_out))) + (L_total*(Tout + 273.15)) - (L_total*T_evp_apply) + R*T_evp_apply*P_c
    B = R*T_evp_apply - L_total/(M_out*1005*(1-BP_out))
    P_total = A/B
    T_cnd_apply = T_cnd_cal_sensible(Tout,L_total,BP_out,Q_out,P_total)
    return P_total, Lsensible_apply, L_total-Lsensible_apply, T_evp_apply-273.15, T_cnd_apply-273.15

class heating_aircon():
    def __init__(self, BP_in,BP_out,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,oneday_testdata, construct_model = True):

        df = pd.read_csv(oneday_testdata)
        print(df.head(2))
        # 实际测试中，如果这三列中有任何一列元素是0，就把这行删掉
        df = df[~((df['Q_sensible'] == 0) | (df['Q_latent'] == 0) | (df['Q_total'] == 0))].reset_index(drop=True)

        # 【--------------------------贝叶斯找BP-----------------------------】
        space = [
            Real(0.1, 0.5, name='BP_in'),
            Real(0.1, 0.5, name='BP_out')

        ]

        # 定义目标函数
        @use_named_args(space)
        def objective(**params):
            BP_in_test = params['BP_in']
            BP_out_test = params['BP_out']

            P_c = cal_Pc(BP_in_test,BP_out_test,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,)
            a,b,c = cal_R_Q_relation(BP_in_test,BP_out_test,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,P_c)

            PUNISHMENT_VALUE = 1e9 
            P_totals = []
            Q_latents = []
            for k in range(len(df)):
                try:
                    if df['volume_setting'][k] == "medium-low":
                        Q_in_apply = 576
                    elif df['volume_setting'][k] == "medium-high":
                        Q_in_apply = 696
                    else:
                        Q_in_apply = 906
                    P_total, Lsensible_apply, L_latent_apply, T_evp_apply, T_cnd_apply = find_P_total(BP_in_test, BP_out_test, a,b,c, P_c, Q_out, df['Q_total'][k],df['Txiru'][k],df['Xxiru'][k] , df['Tout'][k], Q_in_apply)

                except:
                    P_totals.append(PUNISHMENT_VALUE)
                    Q_latents.append(PUNISHMENT_VALUE) # 潜热可以设为 0 或其他占位符
                    continue
                
                if L_latent_apply < 0 or L_latent_apply > df['Q_total'][k] or T_evp_apply > T_cnd_apply:
                    print("出现不符合物理的数值：", "Q_latent_cal, T_evp_apply, T_cnd_apply:", L_latent_apply, T_evp_apply, T_cnd_apply)
                    P_totals.append(PUNISHMENT_VALUE)
                    Q_latents.append(PUNISHMENT_VALUE)
                else:
                    P_totals.append(P_total)
                    Q_latents.append(L_latent_apply)

            # 计算 MSE
            mse = mean_squared_error(P_totals, df['Totalenergy'].tolist()) + 0.08 * mean_squared_error(Q_latents, df['Q_latent'].tolist())
            return float(mse)
        
        if construct_model == True:
            # 运行优化
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
            print(f"BP_in: {result.x[0]}")
            print(f"BP_out: {result.x[1]}")
            print(f"min MSE: {result.fun}")

            self.BP_in = result.x[0]
            self.BP_out = result.x[1]
        
        else:
        #############不乘系数的结果############
            self.BP_in = 0.48087693231060835
            self.BP_out = 0.48087693231060835

        self.Q_out = Q_out
        self.P_c = cal_Pc(self.BP_in,self.BP_out,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,self.Q_out,)
        print("P_c:", self.P_c)
        self.a, self.b, self.c = cal_R_Q_relation(self.BP_in,self.BP_out,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in, self.Q_out,self.P_c)

    def ac_output(self, L_total,Troom,Xroom,Tout, Q_in_apply):
        P_total, Lsensible_apply, L_latent_apply, T_evp_apply, T_cnd_apply = find_P_total(self.BP_in, self.BP_out, self.a,self.b,self.c, self.P_c, self.Q_out, L_total,Troom,Xroom, Tout, Q_in_apply)
        return P_total, Lsensible_apply, L_latent_apply, T_evp_apply, T_cnd_apply


if __name__ == "__main__":
    oneday_test_data = "0701data_processed_0_7.csv"
    ac = heating_aircon(0.25,0.25,500,2200,3300,115,425,960,906,2160, oneday_test_data) # self, BP_in,BP_out,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,Q_in_apply

    test_data_XYplot = "0614_0810_data_processed_0_7.csv"
    df_test1 = pd.read_csv(test_data_XYplot)
    df_test1['Time'] = pd.to_datetime(df_test1['Time'])
    start_time = "2025-08-10 00:00:00"
    end_time = "2025-08-10 23:50:00"
    mask = (df_test1['Time'] >= start_time) & (df_test1['Time'] <= end_time)
    df_test = df_test1.loc[mask].reset_index(drop=True)


    test_data_XYplot = "0614_0810_data_processed_0_7.csv"
    df_test_XYplot = pd.read_csv(test_data_XYplot)
    df_test_XYplot['Time'] = pd.to_datetime(df_test_XYplot['Time'])
    df_test_reheat = df_test_XYplot[df_test_XYplot['mode_real'] == 'reheat'].copy().reset_index(drop=True)
    df_test_cooling = df_test_XYplot[df_test_XYplot['mode_real'] == 'cooling'].copy().reset_index(drop=True)

    # df_test_cooling['Q_sensible'] = 8/7 * df_test_cooling['Q_sensible']
    # df_test_cooling['Q_latent'] = 8/7 * df_test_cooling['Q_latent']
    # df_test_cooling["Q_total"] = 8/7 * df_test_cooling["Q_total"]
    # df_test_cooling["M_in"] = 8/7 * df_test_cooling["M_in"]
 
    P_totals_cooling = []
    Q_latents_cooling = []
    T_evps_cooling = []
    T_cnds_cooling = []

    ###################冷房部分####################
    for e in range(len(df_test_cooling)):
        # print("--------------------------冷房部分--------------------------")
        # print("real energy",df_test_cooling["Totalenergy"][e])
        # if df_test_cooling["Totalenergy"][e] < 10:
        #     P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = 0,0,0,0,0
        if df_test_cooling["volume_setting"][e] == "medium-low":
            # print("冷房小风量")
            P, Lsensible_apply, Q_latent_cal, T_evp_apply, T_cnd_apply = ac.ac_output(df_test_cooling["Q_total"][e],df_test_cooling["Txiru"][e],df_test_cooling["Xxiru"][e],df_test_cooling["Tout"][e], 576) # 冷房小风量：6:00-12:00, 15:00-21:00
        elif df_test_cooling["volume_setting"][e] == "medium-high":
            # print("冷房中风量")
            P, Lsensible_apply, Q_latent_cal, T_evp_apply, T_cnd_apply = ac.ac_output(df_test_cooling["Q_total"][e],df_test_cooling["Txiru"][e],df_test_cooling["Xxiru"][e],df_test_cooling["Tout"][e], 696) # 冷房中风量：6:00-12:00, 15:00-21:00
        elif df_test_cooling["volume_setting"][e] == "very-high":
            # print("冷房大风量")
            P, Lsensible_apply, Q_latent_cal, T_evp_apply, T_cnd_apply = ac.ac_output(df_test_cooling["Q_total"][e],df_test_cooling["Txiru"][e],df_test_cooling["Xxiru"][e],df_test_cooling["Tout"][e], 906) # 冷房大风量：3:00-6:00, 12:00-15:00
        else:
            P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = 0,0,0,0,0
        # print("Calculated energy:", P)
        P_totals_cooling.append(P)
        Q_latents_cooling.append(Q_latent_cal)
        T_evps_cooling.append(T_evp_apply)
        T_cnds_cooling.append(T_cnd_apply)


    def plot_parity(P_totals, y_test):
        plt.figure(figsize=(8, 8))
        # 1. 绘制散点图
        plt.scatter(y_test, P_totals, color='royalblue', alpha=0.6, label='Predicted vs Actual')
        # 2. 绘制 y = x 虚线，范围从 0 到 200
        reference_line = [0, 50]
        plt.plot(reference_line, reference_line, color='red', linestyle='--', linewidth=2, label='Identity Line (y=x)')
        # 3. 强制设置坐标轴范围为 0-200
        # plt.xlim(0, 800)
        # plt.ylim(0, 800)
        # 保持 1:1 的视觉比例，确保 y=x 是完美的 45 度角
        plt.gca().set_aspect('equal', adjustable='box')
        # 图表修饰
        plt.title('EC Prediction Accuracy')
        plt.xlabel('Actual Energy Consumption')
        plt.ylabel('Predicted Energy Consumption')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    plot_parity(P_totals_cooling, df_test_cooling['Totalenergy'].values)
    plot_parity(Q_latents_cooling, df_test_cooling['Q_latent'].values)
    plot_parity(T_evps_cooling, df_test_cooling['Tevp3'].values)
    plot_parity(T_cnds_cooling, df_test_cooling['Tcndout'].values)

    y_true_cool = df_test_cooling['Totalenergy'].values
    rmse_cooling = np.sqrt(mean_squared_error(y_true_cool, P_totals_cooling))
    print(f"EC RMSE cooling: {rmse_cooling:.4f}")

    rmse_cooling_Qlatent = np.sqrt(mean_squared_error(df_test_cooling['Q_latent'].values, Q_latents_cooling))
    print(f"Q_latent RMSE cooling: {rmse_cooling_Qlatent:.4f}")







