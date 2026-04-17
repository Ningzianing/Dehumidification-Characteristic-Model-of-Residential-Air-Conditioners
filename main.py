import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import CoolProp.CoolProp as HAP
import sympy as sp
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.optimize import differential_evolution
from scipy import interpolate
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates

import reheat_dehumidification as rm
import cooling_dehumidification as cm

if __name__ == "__main__":
    experiment_data = r"data\0615data_processed_0_7.csv"
    ac_reheat = rm.reheat_aircon(0.1,0.1,500,2200,3300,115,425,960,906,2160,experiment_data, construct_model= False) #室内機bypass　factor （电中研） , 室外機bypass factor （电中研）, 最小能力、定格能力、最大能力,最小EC,定格EC,最大EC,室内機風量（仕様書）、室外機風量（仕様書）,一日の実験データ
    experiment_data_cooling = r"data\0701data_processed_0_7.csv"
    ac_cool = cm.heating_aircon(0.1,0.1,500,2200,3300,115,425,960,906,2160, experiment_data_cooling, construct_model= False) # self, BP_in,BP_out,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,Q_in_apply

    P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = ac_reheat.output(300, 0.010561504, 22.8846, 23.23546667, 836.4658413, 10)
    print(P, Q_latent_cal)

    P, Lsensible_apply, Q_latent_cal, T_evp_apply, T_cnd_apply = ac_cool.ac_output(848.2075396,22.5294,0.010299202,22.556, 696)
    print(P, Q_latent_cal)


