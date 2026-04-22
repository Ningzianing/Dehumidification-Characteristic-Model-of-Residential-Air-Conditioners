import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

import utils_reheat_dehumidification as reheat_u

class reheat_aircon():
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

        oneday_testdata: csv file containing single-day experimental data. Refer to the '0615data_cooling.csv' file in the 'data' folder

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
        
        # linear relationship between the indoor mass flow rate and heat processed by the indoor unit
        x_data = df['L_total'].values
        y_data = df['M_in'].values
        popt, pcov = curve_fit(reheat_u.Min_Ltotal_function, x_data, y_data, p0=[1.0, 0.0])
        self.Min_a = popt[0] 
        self.Min_b = popt[1] 

        x_fit = np.linspace(x_data.min(), x_data.max(), 100)
        y_fit = reheat_u.Min_Ltotal_function(x_fit, self.Min_a, self.Min_b)
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, color='blue', label='Original Data', alpha=0.6)
        plt.plot(x_fit, y_fit, color='red', linewidth=2)
        plt.title('L_total vs M_in Fitting')
        plt.xlabel('L_total')
        plt.ylabel('M_in')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        # find the calibrated paramters using Bayesian optimization
        space = [
            Real(0, 0.5, name='gamma_cooled'),
            Real(0.1, 0.5, name='BF'),
            Real(0, 0.5, name='gamma_reheat'), 
            Real(0, 5/300, name='delta_t_a'),
            Real(0, 3, name='delta_t_b')
        ]

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
                BF = params['BF']
                gamma_reheat = params['gamma_reheat']
                delta_t_a = params['delta_t_a']
                delta_t_b = params['delta_t_b']

                M_out_solutions = []
                T_evp_solutions = []
                T_cndin_solutions = []
                for k in range(len(df_first_half)):
                    try:
                        omega_evp, T_evp, T_cndin, M_out = reheat_u.solve_Mout(
                            df_first_half['Tout'][k], 
                            df_first_half['winlet'][k], 
                            df_first_half['Tinlet'][k], 
                            df_first_half['L_sensible'][k], 
                            df_first_half['L_latent'][k], 
                            df_first_half['M_in'][k], 
                            df_first_half['Totalenergy'][k], 
                            gamma_cooled, gamma_reheat, BF, delta_t_a, delta_t_b
                        )

                        if np.isinf(M_out) or np.isnan(M_out):
                            return 1e9
                            
                        M_out_solutions.append(M_out)
                        T_evp_solutions.append(T_evp)
                        T_cndin_solutions.append(T_cndin)
                    except:
                        print("solve_Mout error")
                        return 1e9 

                Load_out = (df_first_half["L_sensible"] + df_first_half["L_latent"] + df_first_half["Totalenergy"]).tolist()

                try:
                    popt, pcov = curve_fit(reheat_u.Mout_Loutdoor_function, Load_out, M_out_solutions, p0=[1.0, 0.0])
                    a = popt[0] 
                    b = popt[1] 

                except:
                    return 1e6
                
                P_c = reheat_u.cal_Pc(BF,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out)

                try:
                    poly_func = reheat_u.R_Lcompressor_relation(df_first_half, T_evp_solutions, T_cndin_solutions, gamma_cooled, gamma_reheat, BF, P_c)
                except:
                    return 1e6

                P_totals = []
                L_latents = []

                PUNISHMENT_VALUE = 1e9 

                for e in range(len(df_second_half)):
                    try:
                        P, T_evp_apply, T_cnd_apply, L_latent_cal, SHF = reheat_u.solve_P(
                            200, 
                            df_second_half["winlet"].iloc[e], 
                            df_second_half["Tinlet"].iloc[e], 
                            df_second_half["Tout"].iloc[e], 
                            df_second_half["L_total"].iloc[e], 
                            self.Min_a, self.Min_b, gamma_cooled, gamma_reheat, BF, a, b, poly_func, P_c, delta_t_a, delta_t_b, P_tolerance=15
                        )
                        if L_latent_cal < 0 or L_latent_cal > df_second_half["L_total"].iloc[e] or SHF < 0 or T_evp_apply > T_cnd_apply:
                            print("Physically unrealistic values occurred: ", "L_latent_cal, T_evp_apply, T_cnd_apply:", L_latent_cal, T_evp_apply, T_cnd_apply)
                            P_totals.append(PUNISHMENT_VALUE)
                            L_latents.append(PUNISHMENT_VALUE)
                        else:
                            P_totals.append(P)
                            L_latents.append(L_latent_cal)
                        
                    except Exception as err:
                        print(f"Warning: Row {e} failed to solve. Applying punishment. Error: {err}")

                        P_totals.append(PUNISHMENT_VALUE)
                        L_latents.append(PUNISHMENT_VALUE) 
                        continue

                mse = mean_squared_error(P_totals, df_second_half['Totalenergy'].tolist()) + 0.05 * mean_squared_error(L_latents, df_second_half['L_latent'].tolist())
                mse_all.append(mse)
                i += 1
            return float(sum(mse_all) / len(mse_all))
        
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

            print("Best parameters：")
            print(f"gamma_cooled: {result.x[0]}")
            print(f"BF: {result.x[1]}")
            print(f"gamma_reheat: {result.x[2]}")
            print(f"delta_t_a: {result.x[3]}")
            print(f"delta_t_b: {result.x[4]}")
            print(f"min MSE: {result.fun}")

            self.gamma_cooled = result.x[0]
            self.BF = result.x[1]
            self.gamma_reheat = result.x[2]
            self.delta_t_a = result.x[3]
            self.delta_t_b = result.x[4]
        
        else:
            self.gamma_cooled = 0.43874113146466137
            self.BF = 0.1
            self.gamma_reheat = 0.5
            self.delta_t_a = 0.0
            self.delta_t_b = 3.0

        M_out_solutions_best = []
        T_evp_solutions_best = []
        T_cndin_solutions_best = []
        for k in range(len(df)):
            omega_evp, T_evp, T_cndin, M_out = reheat_u.solve_Mout(df['Tout'][k], df['winlet'][k], df['Tinlet'][k], df['L_sensible'][k], df['L_latent'][k], df['M_in'][k], df['Totalenergy'][k], self.gamma_cooled, self.gamma_reheat, self.BF, self.delta_t_a, self.delta_t_b)
            M_out_solutions_best.append(M_out)
            T_evp_solutions_best.append(T_evp)
            T_cndin_solutions_best.append(T_cndin)

        Load_out_best = (df["L_sensible"] + df["L_latent"] + df["Totalenergy"]).to_list() 

        popt, pcov = curve_fit(reheat_u.Mout_Loutdoor_function, Load_out_best, M_out_solutions_best, p0=[1.0, 0.0])
        self.a = popt[0]
        self.b = popt[1]

        M_out_fit = []
        for i in range(len(df)):
            M_out_fit.append(reheat_u.Mout_Loutdoor_function(Load_out_best[i], self.a, self.b))
        plt.figure(figsize=(10,10)) 
        plt.scatter((df['L_sensible'] + df["L_latent"] + df["Totalenergy"]).to_list(), M_out_solutions_best, color='blue')
        plt.title('Mout-Load_out relation', fontsize=16)
        plt.plot(Load_out_best,M_out_fit, color='red')
        plt.xlabel('Load_out', fontsize=20)
        plt.ylabel('Mout', fontsize=20)  
        plt.grid(True)
        plt.show() 

        self.P_c = reheat_u.cal_Pc(self.BF,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out)
        print("P_c:,", self.P_c)

        self.poly_func = reheat_u.R_Lcompressor_relation(df, T_evp_solutions_best, T_cndin_solutions_best, self.gamma_cooled, self.gamma_reheat, self.BF, self.P_c)
    
    def ac_output(self, P_initial, omega_room, Troom, T_out, L_total, P_tolerance = 15, P_max_iterate = 20, T_evp_tolerance = 0.5, T_evp_max_iterate = 20): 
        """
        P_initial: initial value for total power consumption P_total (W)
        omega_room: absolute humidity of the room air (kg/kgDA)
        Troom:  room air temperature (℃)
        Tout: outdoor air temperature (℃)
        L_total: AC total heat load (W)
        P_tolerance: tolerance value for iteration of P_total
        P_max_iterate: maximum iteration for P_total
        T_evp_tolerance: tolerance value for iteration of T_evp
        T_evp_max_iterate: maximum iteration for T_evp
        """

        P_cal, T_evp, T_cndin, Q_latent, SHF = reheat_u.solve_P(P_initial, omega_room, Troom, T_out, L_total, self.Min_a, self.Min_b,  self.gamma_cooled, self.gamma_reheat, self.BF, self.a, self.b, self.poly_func, self.P_c, self.delta_t_a, self.delta_t_b, P_tolerance, P_max_iterate, T_evp_tolerance, T_evp_max_iterate)
        return P_cal, T_evp, T_cndin, Q_latent, SHF


if __name__ == "__main__":
    experiment_data = r"data\0615data_reheat.csv"
    ac = reheat_aircon(500,2200,3300,115,425,960,906,2160,experiment_data, construct_model= False) 
    P, T_evp_apply, T_cnd_apply, L_latent_cal, SHF = ac.ac_output(250,0.010613295862459, 24.2, 25.1, 759)
    print(P, T_evp_apply, T_cnd_apply, L_latent_cal, SHF)





    


    
    












