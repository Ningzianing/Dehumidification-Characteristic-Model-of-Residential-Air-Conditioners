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

def T_cnd_cal_JIS(Tout,L_total,BF,Q_out,P_total):
    M_out = Q_out / 3600 * 353.25/(Tout+273.15)
    M_cnd = (1 - BF) * M_out
    T_cnd = Tout + (L_total + P_total)/(M_cnd * 1005)
    return T_cnd+273.15

def T_evp_cal_sensible(Troom,L_sensible,BF,Q_in,): ##Q_in: inlet airflow of the indoor unit (m3/h)
    M_in = Q_in / 3600 * 353.25/(Troom+273.15)
    M_evp = (1 - BF) * M_in
    T_evp = Troom - L_sensible/(M_evp * 1005)
    return T_evp+273.15 ##evaporation temperature (K)

def get_w_from_tk(t_K, P):
    # Goff-Gratch formula
    Ps = 10 ** (
        -7.90298 * (373.16 / t_K - 1)
        + 5.02808 * np.log10(373.16 / t_K)
        - 1.3816e-7 * (10 ** (11.344 * (1 - t_K / 373.16)) - 1)
        + 8.1328e-3 * (10 ** (-3.49149 * (373.16 / t_K - 1)) - 1)
        + np.log10(1013.246)
    ) * 100 
    return 0.622 * Ps / (P - Ps)
def get_dew_point(omega, P=101325):  # Given absolute humidity, output dew point temperature
    def objective_min(t, P, target_w):
        try:
            error = get_w_from_tk(t, P) - target_w
            return error**2 
        except:
            return 1e18
    res = minimize_scalar(objective_min, bounds=(253.15, 323.15), args=(P, omega), method='bounded')
    if res.success:
        t_solution = res.x
        residual = res.fun 
    else:
        print("get_dew_point fails to output the minimum value")
        t_solution = np.nan
    return t_solution 

def L_sensible_latent_cal(L_total,Troom,omega_room,BF,Q_in, tolerance=10, max_iterations=20):
    # solving the system of Equations (1)–(4) 
    # Troom, omega_room, L_total, M_in are provided as operation conditions 
    M_in = Q_in / 3600 * 353.25/(Troom+273.15)
    T_evp = T_evp_cal_sensible(Troom,L_total,BF,Q_in) # K

    T_dew_room = get_dew_point(float(omega_room)) # K

    if T_evp >= T_dew_room:
        Lsensible = L_total
        T_evp_final = T_evp #K
    else:
        iteration = 0
        Lsensible = 0.8*L_total 

        while iteration < max_iterations:
            T_evp = T_evp_cal_sensible(Troom,Lsensible,BF,Q_in)
            omega_evp = HAP.HAPropsSI('W', 'T', T_evp, 'P', 101325, 'R', 1.0)
            Llatent = M_in*(1-BF)*2501000*(omega_room-omega_evp)
            Lsensible_cal = L_total - Llatent

            if abs(Lsensible - Lsensible_cal) < tolerance:
                Lsensible = 0.5*(Lsensible + Lsensible_cal)
                T_evp_final = T_evp_cal_sensible(Troom,Lsensible,BF,Q_in) #K
                break

            Lsensible = 0.5*(Lsensible + Lsensible_cal)
            T_evp_final = T_evp_cal_sensible(Troom,Lsensible,BF,Q_in)
            iteration += 1

    return Lsensible, T_evp_final #K

def T_evp_cal_JIS(Troom,RHroom, L_total,BF,Q_in): 
    omega_room = HAP.HAPropsSI('W', 'T', Troom + 273.15, 'R', RHroom/100, 'P', 101325)
    Lsensible, T_evp_final = L_sensible_latent_cal(L_total,Troom,omega_room,BF,Q_in)
    return T_evp_final

def cal_Pc(BF,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,):
    Tout,Troom,RHroom = 35,27,47 # operating conditions reported in specification sheets provided by the manufacturer
    P_c = sp.symbols('P_c')
    R_low = ((L_low / P_low) * L_low / (L_low - (L_low / P_low) * P_c)) / (
                T_evp_cal_JIS(Troom, RHroom, L_low, BF, Q_in) / (
                    T_cnd_cal_JIS(Tout, L_low, BF, Q_out, P_low) - T_evp_cal_JIS(Troom, RHroom, L_low, BF, Q_in)))
    R_medium = ((L_medium / P_medium) * L_medium / (L_medium - (L_medium / P_medium) * P_c)) / (
                T_evp_cal_JIS(Troom, RHroom, L_medium, BF, Q_in) / (
                    T_cnd_cal_JIS(Tout, L_medium, BF, Q_out, P_medium) - T_evp_cal_JIS(Troom, RHroom, L_medium, BF, Q_in)))
    solution = sp.solve(R_low - R_medium, P_c)
    P_c = solution[-1]
    return P_c

def Min_Ltotal_function(x, M_in_a, M_in_b): # indoor mass flow rate is assumed to be linearly related to the heat processed by the indoor unit
    return M_in_a * x + M_in_b

def Mout_Loutdoor_function(x, M_out_a, M_out_b): # outdoor mass flow rate is assumed to be linearly related to the heat processed by the outdoor unit
    return M_out_a * x + M_out_b

def delta_t_P_func(P_total, delta_t_a, delta_t_b): # linear relationship between the total power consumption and the temperature difference between the indoor reheat coil and the outdoor condenser
    val = delta_t_a * P_total + delta_t_b
    return max(0, min(val, 5))

def solve_Mout(T_out, omega_room, Troom, L_sensible, L_latent, M_in, P_total, gamma_cooled, gamma_reheat, BF, delta_t_a, delta_t_b): 
    delta_t = delta_t_P_func(P_total, delta_t_a, delta_t_b) 
    omega_evp = omega_room - L_latent/(M_in*gamma_cooled*(1-BF)*2501000)
    try:
        T_evp = HAP.HAPropsSI('D', 'W', omega_evp, 'P', 101325, 'B', 293.15) - 273.15 
    except:
        T_evp = get_dew_point(float(omega_evp))-273.15

    T_cndin = (M_in*gamma_cooled*(1-BF)*1005*(Troom - T_evp) + M_in*(1-gamma_cooled)*(1-BF)*1005*Troom + M_in*gamma_cooled*gamma_reheat*(1-BF)*1005*T_evp - L_sensible)/(M_in*(1-gamma_cooled)*(1-BF)*1005 + M_in*gamma_cooled*gamma_reheat*(1-BF)*1005)
    M_out = (L_sensible + L_latent + P_total)/(1005*(T_cndin + delta_t -T_out)*(1-BF))
    return omega_evp, T_evp, T_cndin, M_out

def R_Lcompressor_relation(df, T_evp_cals, T_cndin_cals, gamma_cooled, gamma_reheat, BF, P_c):
    L_cnd_cals = []
    L_real_cals = []
    carnoefficients = []
    realefficients = []
    R_dehumid_cals = []

    for y in range(len(df)):
        
        L_cnd_cal = df['M_in'][y] * (1-gamma_cooled) * (1 - BF) * 1005 * (T_cndin_cals[y]-df["Tinlet"][y]) + df['M_in'][y]*gamma_cooled*gamma_reheat*(1-BF)*1005*(T_cndin_cals[y] - T_evp_cals[y])
        L_cnd_cals.append(L_cnd_cal)

        L_real_cal = df["L_sensible"][y] + df["L_latent"][y] + L_cnd_cal 
        L_real_cals.append(L_real_cal)

        carnoefficient = (T_evp_cals[y]+273.15)/(T_cndin_cals[y]-T_evp_cals[y])
        carnoefficients.append(carnoefficient)

        realefficient = L_real_cal/(df["Totalenergy"][y]-P_c) 
        realefficients.append(realefficient)

        R_dehumid_cal = realefficient/carnoefficient
        R_dehumid_cals.append(R_dehumid_cal)

    L_real_cals_np = np.array(L_real_cals,dtype = float)
    R_dehumid_cals_np = np.array(R_dehumid_cals, dtype = float)
    coeffs = np.polyfit(L_real_cals_np, R_dehumid_cals_np, deg=2)

    print(coeffs)
    poly_func = np.poly1d(coeffs)
    return poly_func

def iterate_T_evp(T_evp, M_in, gamma_cooled, gamma_reheat, BF, omega_room, Troom, T_cndin, L_total, P=101325):
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
    omega_evp = 0.622 * Ps / (P - Ps)
    
    L_latent_cal = M_in * gamma_cooled * (1 - BF) * 2501000 * (omega_room - omega_evp)
    L_sensible_cal = L_total-L_latent_cal
    T_evp_cal = (L_sensible_cal - M_in * gamma_cooled * (1 - BF) * 1005 * Troom + M_in * (1 - gamma_cooled) * (1 - BF) * 1005 * (T_cndin - Troom) + M_in * gamma_cooled * gamma_reheat * (1-BF) * 1005 * T_cndin) / (M_in * gamma_cooled * gamma_reheat * (1-BF) * 1005 - M_in * gamma_cooled * (1-BF) * 1005)
    return float(T_evp_cal)

def solve_main_eqs(M_in, omega_room, Troom, T_out, L_total, P_aircon, gamma_cooled, gamma_reheat, BF, a, b, delta_t_a, delta_t_b, 
                        T_evp_tolerance=0.5, T_evp_max_iterate=20): # solve the equation to find omega_evp, T_evp, T_cndin, Q_latent
    M_out = a * (L_total + P_aircon) + b
    delta_t = delta_t_P_func(P_aircon, delta_t_a, delta_t_b)
    T_cndin = (L_total + P_aircon)/((1-BF)*M_out*1005) + T_out - delta_t

    continue_T_evp = True
    T_evp_solution = 10
    T_evp_iterate= 0
    T_evp_max = 20
    T_evp_min = 0
    while continue_T_evp == True and T_evp_iterate <= T_evp_max_iterate:
        T_evp_solution_cal = iterate_T_evp(T_evp_solution, M_in, gamma_cooled, gamma_reheat, BF, omega_room, Troom, T_cndin, L_total)
        T_evp_solution_cal = max(min(T_evp_solution_cal, T_evp_max), T_evp_min)
        if abs(T_evp_solution - T_evp_solution_cal) <= T_evp_tolerance:
            T_evp_solution = (T_evp_solution + T_evp_solution_cal) / 2
            continue_T_evp = False       
        else:
            T_evp_solution = (T_evp_solution + T_evp_solution_cal) / 2
            T_evp_iterate += 1    
    if T_evp_iterate==T_evp_max_iterate:
        print("P_aircon:", P_aircon, "L_total:", L_total, "gamma_cooled, gamma_reheat, BF:", gamma_cooled, gamma_reheat, BF, "solve_main_eqs does not converge" )

    T_evp_solution_K = T_evp_solution + 273.15
    Ps_solution = 10 ** (
        -7.90298 * (373.16 / T_evp_solution_K - 1)
        + 5.02808 * np.log10(373.16 / T_evp_solution_K)
        - 1.3816e-7 * (10 ** (11.344 * (1 - T_evp_solution_K / 373.16)) - 1)
        + 8.1328e-3 * (10 ** (-3.49149 * (373.16 / T_evp_solution_K - 1)) - 1)
        + np.log10(1013.246)
    ) * 100 

    omega_evp_solution = 0.622 * Ps_solution / (101325 - Ps_solution)
    
    L_latent_solution = M_in*gamma_cooled*(1-BF)*2501000*(omega_room-omega_evp_solution)
    return omega_evp_solution, T_evp_solution, T_cndin, L_latent_solution

def solve_P(P_initial, omega_room, Troom, T_out, L_total, Min_a, Min_b, gamma_cooled, gamma_reheat, BF, a, b, poly_func, P_c, delta_t_a, delta_t_b, P_tolerance = 15, P_max_iterate = 20, T_evp_tolerance = 0.5, T_evp_max_iterate = 20): 
    M_in = Min_Ltotal_function(L_total, Min_a, Min_b)

    continuecal = True
    P = P_initial
    iterate = 0

    P_min = 100 
    P_max = 400  

    while continuecal == True:
        omega_evp, T_evp, T_cndin, L_latent = solve_main_eqs(M_in, omega_room, Troom, T_out, L_total, P, gamma_cooled, gamma_reheat, BF, a, b, delta_t_a, delta_t_b, T_evp_tolerance, T_evp_max_iterate) 
        L_cndin = M_in*(1-gamma_cooled)*(1-BF)*1005*(T_cndin - Troom) + M_in*gamma_cooled*gamma_reheat*(1-BF)*1005*(T_cndin-T_evp)

        L_total_compressor = L_total + L_cndin
        R_dehumid = poly_func(L_total_compressor)
        P_total = (L_total_compressor / ((T_evp+273.15)*R_dehumid/(T_cndin-T_evp))) + P_c
        P_total = max(min(P_total, P_max), P_min)

        if abs(P-P_total) < P_tolerance or iterate > P_max_iterate:
            continuecal = False
            break
        else:
            P_new = (P + P_total)/2
            P = max(min(P_new, P_max), P_min)
            iterate += 1
        if iterate >= P_max_iterate:
            print("L_total:", L_total, "gamma_cooled, gamma_reheat, BF:", gamma_cooled, gamma_reheat, BF, "solve_P does not converge")

    return max(min((P + P_total)/2, P_max), P_min), T_evp, T_cndin, L_latent, (L_total - L_latent) / L_total 






    


    
    












