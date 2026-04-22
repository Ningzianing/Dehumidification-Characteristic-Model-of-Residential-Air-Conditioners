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

import utils_cooling_dehumidification as cooling_u

class cooling_aircon():
    def __init__(self, L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,oneday_testdata, construct_model = True):
        """
        L_low: minimum capacity of the air conditioner (W)
        L_medium: rated capacity of the air conditioner (W)
        L_high: maximum capacity of the air conditioner (W)
        P_low: power consumption under minimum capacity (W)
        P_medium: power consumption under medium capacity (W)
        P_high: power consumption under maximum capacity (W)
        Q_in: 'High' airflow rate for the indoor unit (m3/h)
        Q_out: 'High' airflow rate for the outdoor unit (m3/h)

        The above parameters can all be found in manufacturer-provided performance data.

        oneday_testdata: csv file containing single-day experimental data. Refer to the '0701data_cooling.csv' file in the 'data' folder

        construct_model: Through our experiments, we have determined a calibrated parameter specifically for our experimental air conditioning system. If you set this parameter to False, the system will default to this pre-calibrated value. However, please note that this value may NOT be applicable to other air conditioning systems.
        """

        df = pd.read_csv(oneday_testdata)
        print(df.head(2))

        df["L_latent"] = df["M_in"] * 2501000 * (df['winlet'] - df['woutlet']) 
        df.loc[df['L_latent'] < 0, 'L_latent'] = 0
        df["L_sensible"] = df["M_in"] * 1005 * (df['Tinlet'] - df['Toutlet']) 
        df.loc[df['L_sensible'] < 0, 'L_sensible'] = 0
        df["L_total"] = df["L_sensible"] + df["L_latent"]

        df = df[~((df['L_total'] == 0))].reset_index(drop=True)
        
        # determine the optimal value of the calibrated value BF using Bayesian optimization
        space = [
            Real(0.1, 0.5, name='BF')
        ]

        @use_named_args(space)
        def objective(**params): # opjective function
            BF_test = params['BF']

            P_c = cooling_u.cal_Pc(BF_test,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,)
            a,b,c = cooling_u.cal_R_Ltotal_relation(BF_test,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,P_c)

            PUNISHMENT_VALUE = 1e9 
            P_totals = []
            L_latents = []
            for k in range(len(df)):
                try:
                    if df['volume_setting'][k] == "medium-low":
                        Q_in_apply = 576
                    elif df['volume_setting'][k] == "medium-high":
                        Q_in_apply = 696
                    else:
                        Q_in_apply = 906
                    P_total, Lsensible_apply, L_latent_apply, T_evp_apply, T_cnd_apply = cooling_u.find_P_total(BF_test, a,b,c, P_c, Q_out, df['L_total'][k],df['Tinlet'][k],df['winlet'][k] , df['Tout'][k], Q_in_apply)

                except:
                    P_totals.append(PUNISHMENT_VALUE)
                    L_latents.append(PUNISHMENT_VALUE) 
                    continue
                
                if L_latent_apply < 0 or L_latent_apply > df['L_total'][k] or T_evp_apply > T_cnd_apply:
                    print("Physically unrealistic values occurred: ", "L_latent_apply, T_evp_apply, T_cnd_apply:", L_latent_apply, T_evp_apply, T_cnd_apply)
                    P_totals.append(PUNISHMENT_VALUE)
                    L_latents.append(PUNISHMENT_VALUE)
                else:
                    P_totals.append(P_total)
                    L_latents.append(L_latent_apply)

            mse = mean_squared_error(P_totals, df['Totalenergy'].tolist()) + 0.08 * mean_squared_error(L_latents, df['L_latent'].tolist())
            return float(mse)
        
        if construct_model == True:
            result = gp_minimize(
                objective,
                space,
                n_calls=100,             
                n_random_starts=5,     
                n_jobs=-1,          
                noise=1e-10,        
                acq_func='EI', 
                verbose=True
            )

            print("Best parameters: ")
            print(f"BF: {result.x[0]}")
            print(f"min MSE: {result.fun}")

            self.BF = result.x[0]
        
        else:
        # BF determined for our experimental air conditioning system
            self.BF = 0.48087693231060835

        self.Q_out = Q_out
        self.P_c = cooling_u.cal_Pc(self.BF,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,self.Q_out,)
        print("P_c:", self.P_c)
        self.a, self.b, self.c = cooling_u.cal_R_Ltotal_relation(self.BF,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in, self.Q_out,self.P_c)
        print("R_Ltotal_relation:", self.a, self.b, self.c)

    def ac_output(self, L_total,Troom,omega_room,Tout, Q_in_apply):
        """
        L_total: AC total heat load (W)
        Troom:  room air temperature (℃)
        omega_room: absolute humidity of the room air (kg/kgDA)
        Tout: outdoor air temperature (℃)
        Q_in_apply: airflow rate under current operating condition (m3/h)
        """

        P_total, Lsensible_apply, L_latent_apply, T_evp_apply, T_cnd_apply = cooling_u.find_P_total(self.BF, self.a,self.b,self.c, self.P_c, self.Q_out, L_total,Troom,omega_room, Tout, Q_in_apply)
        return P_total, Lsensible_apply, L_latent_apply, T_evp_apply, T_cnd_apply

if __name__ == "__main__":
    # construct the model 
    oneday_test_data = r"data\0701data_cooling.csv"
    ac = cooling_aircon(500,2200,3300,115,425,960,906,2160, oneday_test_data, construct_model=False) 

    P, Lsensible_apply, Q_latent_cal, T_evp_apply, T_cnd_apply = ac.ac_output(476, 25.458, 0.015851711, 28.65173333, 576) 

    print(P, Lsensible_apply, Q_latent_cal, T_evp_apply, T_cnd_apply)







