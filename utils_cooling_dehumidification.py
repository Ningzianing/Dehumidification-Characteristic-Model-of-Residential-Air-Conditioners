import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import CoolProp.CoolProp as HAP
import sympy as sp
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize_scalar

def T_evp_cal_sensible(Troom,L_sensible,BF,Q_in,): ##Q_in: inlet airflow of the indoor unit (m3/h)
    M_in = Q_in / 3600 * 353.25/(Troom+273.15)
    M_evp = (1 - BF) * M_in
    T_evp = Troom - L_sensible/(M_evp * 1005)
    return T_evp+273.15 ##evaporation temperature (K)

def T_cnd_cal_sensible(Tout,L_total,BF,Q_out,P_total): # Q_out: inlet airflow of the outdoor unit (m3/h)
    M_out = Q_out / 3600 * (353.25/(Tout+273.15))
    M_cnd = (1 - BF) * M_out
    T_cnd = Tout + (L_total + P_total)/(M_cnd * 1005)
    return T_cnd + 273.15  ##condensation temperature (K)

def quadratic_function(x, a, b, c): # quadratic regression to model 𝑅 as a function of the total cooling load
    return a * x ** 2 + b * x + c

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
    def objective_min(t, P, omega):
        try:
            error = get_w_from_tk(t, P) - omega
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

def T_evp_cal_total(Troom,RHroom, L_total, BF ,Q_in):
    omega_room = HAP.HAPropsSI('W', 'T', Troom + 273.15, 'R', RHroom/100, 'P', 101325)
    Lsensible, T_evp_final = L_sensible_latent_cal(L_total,Troom,omega_room,BF,Q_in)
    return T_evp_final

def cal_Pc(BF,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,):
    Tout,Troom,RHroom = 35,27,47 # operating conditions reported in specification sheets provided by the manufacturer
    P_c = sp.symbols('P_c')
    R_low = ((L_low / P_low) * L_low / (L_low - (L_low / P_low) * P_c)) / (
                T_evp_cal_total(Troom, RHroom, L_low, BF, Q_in) / (
                    T_cnd_cal_sensible(Tout, L_low, BF, Q_out, P_low) - T_evp_cal_total(Troom, RHroom, L_low, BF, Q_in)))
    R_medium = ((L_medium / P_medium) * L_medium / (L_medium - (L_medium / P_medium) * P_c)) / (
                T_evp_cal_total(Troom, RHroom, L_medium, BF, Q_in) / (
                    T_cnd_cal_sensible(Tout, L_medium, BF, Q_out, P_medium) - T_evp_cal_total(Troom, RHroom, L_medium, BF, Q_in)))
    solution = sp.solve(R_low - R_medium, P_c)
    P_c = solution[-1]
    return P_c

def cal_R_Ltotal_relation(BF,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,P_c): 
    Tout,Troom,RHroom = 35,27,47 # operating conditions reported in specification sheets provided by the manufacturer
    R_low    = ((L_low/P_low) *L_low /(L_low-(L_low/P_low)* P_c))  / (T_evp_cal_total(Troom,RHroom, L_low,BF,Q_in)/(T_cnd_cal_sensible(Tout,L_low,BF,Q_out,P_low) - T_evp_cal_total(Troom,RHroom,L_low,BF,Q_in)))
    R_medium = ((L_medium/P_medium)*L_medium/(L_medium-(L_medium/P_medium)* P_c)) / (T_evp_cal_total(Troom,RHroom,L_medium,BF,Q_in)/(T_cnd_cal_sensible(Tout,L_medium,BF,Q_out,P_medium)- T_evp_cal_total(Troom,RHroom,L_medium,BF,Q_in)))
    R_high   = ((L_high/P_high)    *L_high  /(L_high-(L_high/P_high)* P_c)) / (T_evp_cal_total(Troom,RHroom,L_high,BF,Q_in)/(T_cnd_cal_sensible(Tout,L_high,BF,Q_out,P_high) - T_evp_cal_total(Troom,RHroom,L_high,BF,Q_in)))

    x_data = [L_low,L_medium,L_high]
    y_data = [R_low,R_medium,R_high]

    params, covariance = curve_fit(quadratic_function, x_data, y_data)

    a, b, c = params
    return a,b,c

def find_P_total(BF, a,b,c, P_c, Q_out, L_total,Troom,omega_room,Tout, Q_in_apply):
    M_in = Q_in_apply / 3600 * 353.25/(Troom+273.15)
    M_out = Q_out / 3600 * 353.25/(Tout+273.15)

    # solving the system of Equations (1)–(4)
    Lsensible_apply, T_evp_apply = L_sensible_latent_cal(L_total,Troom,omega_room,BF, Q_in_apply, tolerance=5, max_iterations=20) 

    R = quadratic_function(L_total, a, b, c)

    # Apply the method of elimination to solve Equations (5) and (11)
    A = (L_total**2/(M_out*1005*(1-BF))) + (L_total*(Tout + 273.15)) - (L_total*T_evp_apply) + R*T_evp_apply*P_c
    B = R*T_evp_apply - L_total/(M_out*1005*(1-BF))
    P_total = A/B 
    T_cnd_apply = T_cnd_cal_sensible(Tout,L_total,BF,Q_out,P_total)

    return P_total, Lsensible_apply, L_total-Lsensible_apply, T_evp_apply-273.15, T_cnd_apply-273.15







