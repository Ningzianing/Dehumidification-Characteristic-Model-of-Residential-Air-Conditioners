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
import matplotlib.dates as mdates

import reheat_dehumidification as rm
import cooling_dehumidification as cm

def plot_onedayfitting(df_test, P_lst,legend_loc):
    L_sensible_array = np.array(df_test['L_sensible'].values, dtype = float)
    L_latent_array = np.array(df_test['L_latent'].values, dtype = float)
    P_measured = df_test['Totalenergy'].copy()

    fig, ax1 = plt.subplots(figsize=(18, 5))
    ax2 = ax1.twinx()
    width = 0.0055
    ax2.bar(df_test['Time'], L_latent_array, width=width, 
            label='measured latent heat', color='skyblue', alpha=0.39)
    ax2.bar(df_test['Time'], L_sensible_array, width=width, 
            bottom=L_latent_array, label='measured sensible heat', color='lightcoral', alpha=0.39)

    ax2.set_ylabel('Total Heat (W)', fontsize=18)
    # ax2.legend(loc='lower right', fontsize=16)
    ax2.tick_params(axis='y', which='major', labelsize=14)
    ax2.set_ylim(0,2000)

    line_zorder = 10 
    ax1.plot(df_test['Time'], P_measured, label='Measured', 
            color='black', linewidth=2.2, zorder=line_zorder) 
    ax1.plot(df_test['Time'], P_lst, label='M.3 (Proposed)', 
            color='red', linewidth=2.2, zorder=line_zorder)
    ax1.set_xlabel('Time (hh:mm)', fontsize=18)
    ax1.set_ylabel('Power Consumption (W)', fontsize=18)
    ax1.legend(loc='lower left', fontsize=16)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False) 
    ax1.set_ylim(0,500)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.grid(True, linestyle='--', alpha=0.3, zorder=0) 
    ax1.tick_params(axis='both', which='major', labelsize=14)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    ax1.legend(all_handles, all_labels, 
               loc=legend_loc, 
               ncol=2, 
               fontsize=16,
               columnspacing=1.0, 
               handletextpad=0.5, 
               frameon=True)
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.show()

def plot_onedayfitting_latent(df_test, latent_lst,legend_loc):
    L_sensible_array = np.array(df_test['L_sensible'].values, dtype = float)
    L_latent_array = np.array(df_test['L_latent'].values, dtype = float)
    L_latent_measured = df_test['L_latent'].copy()
    fig, ax1 = plt.subplots(figsize=(18, 5))
    width = 0.0055
    line_zorder = 10 
    ax1.bar(df_test['Time'], L_latent_array, width=width, 
            label='measured latent heat', color='skyblue', alpha=0.39, zorder=1)
    ax1.bar(df_test['Time'], L_sensible_array, width=width, 
            bottom=L_latent_array, label='measured sensible heat', color='lightcoral', alpha=0.39, zorder=1)
    ax1.plot(df_test['Time'], L_latent_measured, label='Measured', 
            color='black', linewidth=2.2, zorder=line_zorder)
    ax1.plot(df_test['Time'], latent_lst, label='M.3 (Proposed)', 
            color='red', linewidth=2.2, zorder=line_zorder)
    ax1.set_xlabel('Time (hh:mm)', fontsize=18)
    ax1.set_ylabel('Heat (W)', fontsize=18) 
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.grid(True, linestyle='--', alpha=0.3, zorder=0) 
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_ylim(0,2000)
    ax1.legend(loc = legend_loc, fontsize=16, ncol=2) 
    plt.tight_layout()
    plt.show()

experiment_data = r"data\0615data_reheat.csv"
ac_reheat = rm.reheat_aircon(500,2200,3300,115,425,960,906,2160,experiment_data, construct_model= False) 
experiment_data_cooling = r"data\0701data_cooling.csv"
ac_cool = cm.cooling_aircon(500,2200,3300,115,425,960,906,2160, experiment_data_cooling, construct_model= False) 

test_data_2month = r"data\0614_0810_data.csv"
df_test1 = pd.read_csv(test_data_2month)
df_test1['Time'] = pd.to_datetime(df_test1['Time'])
df_test1["L_latent"] = df_test1["M_in"] * 2501000 * (df_test1['winlet'] - df_test1['woutlet']) 
df_test1.loc[df_test1['L_latent'] < 0, 'L_latent'] = 0
df_test1["L_sensible"] = df_test1["M_in"] * 1005 * (df_test1['Tinlet'] - df_test1['Toutlet']) 
df_test1.loc[df_test1['L_sensible'] < 0, 'L_sensible'] = 0
df_test1["L_total"] = df_test1["L_sensible"] + df_test1["L_latent"]

##################################
# evaluate the calibrated model parameters on June 15
##################################
start_time = "2025-06-15 00:00:00"
end_time = "2025-06-15 23:50:00"
mask = (df_test1['Time'] >= start_time) & (df_test1['Time'] <= end_time)
df_test_reheat_ftting = df_test1.loc[mask].reset_index(drop=True)

P_totals_0615 = []
L_latents_0615 = []

for e in range(len(df_test_reheat_ftting)):
    print("--------------------------------------------------------------")
    print("real energy",df_test_reheat_ftting["Totalenergy"][e],e)
    P, T_evp_apply, T_cnd_apply, L_latent_cal, SHF = ac_reheat.ac_output(300, df_test_reheat_ftting["winlet"][e], df_test_reheat_ftting["Tinlet"][e], df_test_reheat_ftting["Tout"][e], df_test_reheat_ftting["L_total"][e], P_tolerance=10)

    print("Calculated energy:", P)
    P_totals_0615.append(P)
    L_latents_0615.append(L_latent_cal)

##################################
# evaluate the calibrated model parameter on July 1
##################################
start_time = "2025-07-01 00:00:00"
end_time = "2025-07-01 23:50:00"
mask = (df_test1['Time'] >= start_time) & (df_test1['Time'] <= end_time)
df_test_cooling_ftting = df_test1.loc[mask].reset_index(drop=True)

P_totals_0701 = []
L_latents_0701 = []

for e in range(len(df_test_cooling_ftting)):
    print("--------------------------------------------------------------")
    print("real energy",df_test_cooling_ftting["Totalenergy"][e],e)
    P, Lsensible_apply, L_latent_cal, T_evp_apply, T_cnd_apply = ac_cool.ac_output(df_test_cooling_ftting["L_total"][e],df_test_cooling_ftting["Tinlet"][e],df_test_cooling_ftting["winlet"][e],df_test_cooling_ftting["Tout"][e], 696)
        
    print("Calculated energy:", P)
    P_totals_0701.append(P)
    L_latents_0701.append(L_latent_cal)

plot_onedayfitting(df_test_reheat_ftting, P_totals_0615, "upper left")
plot_onedayfitting(df_test_cooling_ftting, P_totals_0701, "upper center")

plot_onedayfitting_latent(df_test_reheat_ftting, L_latents_0615, "upper left")
plot_onedayfitting_latent(df_test_cooling_ftting, L_latents_0701, "upper center")

##################################
# two month evaluation
##################################
def plot_parity(P_totals, y_test, color, y_upper):
    plt.figure(figsize=(15, 15))
    plt.scatter(y_test, P_totals, color=color, alpha=0.8, s=100)
    reference_line = [0, y_upper]
    plt.plot(reference_line, reference_line, color='grey', linestyle='--', linewidth=2.2)
    plt.xlim(0, y_upper)
    plt.ylim(0, y_upper)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Measured', fontsize = 18)
    plt.ylabel('Predicted', fontsize = 18)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

df_test_reheat = df_test1[df_test1['mode'] == 'reheat'].copy().reset_index(drop=True)
df_test_cooling = df_test1[df_test1['mode'] == 'cooling'].copy().reset_index(drop=True)
y_test_reheat_P = df_test_reheat['Totalenergy']
y_test_cooling_P = df_test_cooling['Totalenergy']
y_test_reheat_latent = df_test_reheat['L_latent']
y_test_cooling_latent = df_test_cooling['L_latent']

P_totals_reheat = []   
P_totals_cooling = []
L_latents_reheat = []
L_latents_cooling = []

for e in range(len(df_test_reheat)):
    print("-------------------------reheat dehumidification---------------------------")
    print("real energy",df_test_reheat["Totalenergy"][e])
    P, T_evp_apply, T_cnd_apply, L_latent_cal, SHF = ac_reheat.ac_output(300, df_test_reheat["winlet"][e], df_test_reheat["Tinlet"][e], df_test_reheat["Tout"][e], df_test_reheat["L_total"][e], P_tolerance=10)
    print("Calculated energy:", P)
    P_totals_reheat.append(max(0,P))
    L_latents_reheat.append(max(0,L_latent_cal))

for e in range(len(df_test_cooling)):
    print("--------------------------cooling dehumidification--------------------------")
    print("real energy",df_test_cooling["Totalenergy"][e])
    if df_test_cooling["volume_setting"][e] == "medium-low":
        P, Lsensible_apply, L_latent_cal, T_evp_apply, T_cnd_apply = ac_cool.ac_output(df_test_cooling["L_total"][e],df_test_cooling["Tinlet"][e],df_test_cooling["winlet"][e],df_test_cooling["Tout"][e], 576) 
    elif df_test_cooling["volume_setting"][e] == "medium-high":
        P, Lsensible_apply, L_latent_cal, T_evp_apply, T_cnd_apply = ac_cool.ac_output(df_test_cooling["L_total"][e],df_test_cooling["Tinlet"][e],df_test_cooling["winlet"][e],df_test_cooling["Tout"][e], 696)
    elif df_test_cooling["volume_setting"][e] == "very-high":
        P, Lsensible_apply, L_latent_cal, T_evp_apply, T_cnd_apply = ac_cool.ac_output(df_test_cooling["L_total"][e],df_test_cooling["Tinlet"][e],df_test_cooling["winlet"][e],df_test_cooling["Tout"][e], 906) 
    else:
        P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = 0,0,0,0,0
    print("Calculated energy:", P)
    P_totals_cooling.append(P)
    L_latents_cooling.append(L_latent_cal)

plot_parity(P_totals_reheat, y_test_reheat_P, 'red', 400)
plot_parity(P_totals_cooling, y_test_cooling_P, 'red', 800)
plot_parity(L_latents_reheat, y_test_reheat_latent, 'red', 1500)
plot_parity(L_latents_cooling, y_test_cooling_latent, 'red', 1000)

