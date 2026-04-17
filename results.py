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

import C_good_reheat_model_packpack_faster_for_paper_1 as rm
import C_good_cooling_model_pack_for_paper as cm

def plot_onedayfitting(df_test, P_lst,legend_loc):
    Q_sensible_array = np.array(df_test['Q_sensible'].values, dtype = float)
    Q_latent_array = np.array(df_test['Q_latent'].values, dtype = float)
    P_measured = df_test['Totalenergy'].copy()

    fig, ax1 = plt.subplots(figsize=(18, 5))
    ax2 = ax1.twinx()
    width = 0.0055
    ax2.bar(df_test['Time'], Q_latent_array, width=width, 
            label='measured latent heat', color='skyblue', alpha=0.39)
    ax2.bar(df_test['Time'], Q_sensible_array, width=width, 
            bottom=Q_latent_array, label='measured sensible heat', color='lightcoral', alpha=0.39)

    ax2.set_ylabel('Total Heat (W)', fontsize=18)
    # ax2.legend(loc='lower right', fontsize=16)
    ax2.tick_params(axis='y', which='major', labelsize=14)
    ax2.set_ylim(0,2000)

    line_zorder = 10 
    ax1.plot(df_test['Time'], P_measured, label='Measured', 
            color='black', linewidth=2.2, zorder=line_zorder) # 稍微加粗Measured
    ax1.plot(df_test['Time'], P_lst, label='M.3 (Proposed)', 
            color='red', linewidth=2.2, zorder=line_zorder)
    ax1.set_xlabel('Time (hh:mm)', fontsize=18)
    ax1.set_ylabel('Power Consumption (W)', fontsize=18)
    ax1.legend(loc='lower left', fontsize=16)
    # --- 3. 核心步骤：调整图层顺序和透明度 ---
    # 将 ax1 (线条) 的层级放到 ax2 (柱子) 之上
    ax1.set_zorder(ax2.get_zorder() + 1)
    # **非常重要**：必须将 ax1 的背景设置为透明，否则 ax2 会被完全遮挡
    ax1.patch.set_visible(False) 
    ax1.set_ylim(0,500)
    # --- 4. 格式化时间轴 ---
    # 注意：所有的格式化必须在 ax1 上进行，因为它现在是顶层
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # plt.title('Measured vs Calculated Energy with Q Components', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.3, zorder=0) 
    ax1.tick_params(axis='both', which='major', labelsize=14)


    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # 2. 合并它们
    # 这样合并后，第一列是 Line (Measured/Calculated)，第二列是 Bar (Latent/Sensible)
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    ax1.legend(all_handles, all_labels, 
               loc=legend_loc, 
               ncol=2, 
               fontsize=16,
               columnspacing=1.0, # 调整左右两列之间的间距
               handletextpad=0.5, # 调整图标与文字之间的间距
               frameon=True)
    # --- 调整整体边距以容纳两行图例 ---
    plt.subplots_adjust(top=0.8) # 这里的 0.8 根据实际效果微调，数值越小上方留白越多
    plt.tight_layout()
    plt.show()

def plot_onedayfitting_latent(df_test, latent_lst,legend_loc):
    Q_sensible_array = np.array(df_test['Q_sensible'].values, dtype = float)
    Q_latent_array = np.array(df_test['Q_latent'].values, dtype = float)
    Q_latent_measured = df_test['Q_latent'].copy()

    fig, ax1 = plt.subplots(figsize=(18, 5))
    width = 0.0055
    # 线条的层级，确保在柱子上方
    line_zorder = 10 
    # --- 2. 绘制堆叠柱状图 (全部画在 ax1 上) ---
    ax1.bar(df_test['Time'], Q_latent_array, width=width, 
            label='measured latent heat', color='skyblue', alpha=0.39, zorder=1)
    ax1.bar(df_test['Time'], Q_sensible_array, width=width, 
            bottom=Q_latent_array, label='measured sensible heat', color='lightcoral', alpha=0.39, zorder=1)
    # --- 3. 绘制折线图 (也画在 ax1 上) ---
    ax1.plot(df_test['Time'], Q_latent_measured, label='Measured', 
            color='black', linewidth=2.2, zorder=line_zorder)
    ax1.plot(df_test['Time'], latent_lst, label='M.3 (Proposed)', 
            color='red', linewidth=2.2, zorder=line_zorder)
    # --- 4. 设置坐标轴标签和格式 ---
    ax1.set_xlabel('Time (hh:mm)', fontsize=18)
    ax1.set_ylabel('Heat (W)', fontsize=18) # 统一标签名
    # 格式化时间轴
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # 设置网格和刻度
    ax1.grid(True, linestyle='--', alpha=0.3, zorder=0) 
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_ylim(0,2000)
    # --- 5. 合并图例 ---
    # 因为都在 ax1 上，直接调用一次 legend 即可包含所有 label
    ax1.legend(loc = legend_loc, fontsize=16, ncol=2) # ncol=2 可以让图例排成两列，节省空间
    plt.tight_layout()
    plt.show()


experiment_data = r"data\0615data_processed_0_7.csv"
ac_reheat = rm.reheat_aircon(0.1,0.1,500,2200,3300,115,425,960,906,2160,experiment_data, construct_model= False) #室内機bypass　factor （电中研） , 室外機bypass factor （电中研）, 最小能力、定格能力、最大能力,最小EC,定格EC,最大EC,室内機風量（仕様書）、室外機風量（仕様書）,一日の実験データ
experiment_data_cooling = r"data\0701data_processed_0_7.csv"
ac_cool = cm.heating_aircon(0.1,0.1,500,2200,3300,115,425,960,906,2160, experiment_data_cooling, construct_model= False) # self, BP_in,BP_out,L_low,L_medium,L_high,P_low,P_medium,P_high,Q_in,Q_out,Q_in_apply


test_data_XYplot = r"data\0614_0810_data_processed_0_7.csv"
df_test1 = pd.read_csv(test_data_XYplot)
df_test1['Time'] = pd.to_datetime(df_test1['Time'])

##################################
# 0615 fitting效果验证
##################################
start_time = "2025-06-15 00:00:00"
end_time = "2025-06-15 23:50:00"
mask = (df_test1['Time'] >= start_time) & (df_test1['Time'] <= end_time)
df_test_reheat_ftting = df_test1.loc[mask].reset_index(drop=True)

P_totals_0615 = []
Q_latents_0615 = []

for e in range(len(df_test_reheat_ftting)):
    print("--------------------------------------------------------------")
    print("real energy",df_test_reheat_ftting["Totalenergy"][e],e)
    # if df_test["Totalenergy"][e] < 10:
    #     P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = 0,0,0,0,0
    time_thes = pd.to_datetime("2025-06-15 13:30:00")
    if df_test_reheat_ftting['Time'][e] <= time_thes:
        P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = ac_reheat.output(300, df_test_reheat_ftting["Xxiru"][e], df_test_reheat_ftting["Txiru"][e], df_test_reheat_ftting["Tout"][e], (df_test_reheat_ftting["Q_sensible"][e] + df_test_reheat_ftting["Q_latent"][e]), 10)
    else:
        P, Lsensible_apply, Q_latent_cal, T_evp_apply, T_cnd_apply = 0,0,0,0,0 # 冷房大风量：3:00-6:00, 12:00-15:00
        
    print("Calculated energy:", P)
    P_totals_0615.append(P)
    Q_latents_0615.append(Q_latent_cal)

##################################
# 0701 fitting效果验证
##################################
start_time = "2025-07-01 00:00:00"
end_time = "2025-07-01 23:50:00"
mask = (df_test1['Time'] >= start_time) & (df_test1['Time'] <= end_time)
df_test_cooling_ftting = df_test1.loc[mask].reset_index(drop=True)

P_totals_0701 = []
Q_latents_0701 = []

for e in range(len(df_test_cooling_ftting)):
    print("--------------------------------------------------------------")
    print("real energy",df_test_cooling_ftting["Totalenergy"][e],e)
    # if df_test["Totalenergy"][e] < 10:
    #     P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = 0,0,0,0,0
    time_tres1 = pd.to_datetime("2025-07-01 06:00:00")
    time_tres2 = pd.to_datetime("2025-07-01 17:50:00")
    if df_test_cooling_ftting['Time'][e] >= time_tres1 and df_test_cooling_ftting['Time'][e] <= time_tres2:
        P, Lsensible_apply, Q_latent_cal, T_evp_apply, T_cnd_apply = ac_cool.ac_output(df_test_cooling_ftting["Q_total"][e],df_test_cooling_ftting["Txiru"][e],df_test_cooling_ftting["Xxiru"][e],df_test_cooling_ftting["Tout"][e], 696)
    else:
        P, Lsensible_apply, Q_latent_cal, T_evp_apply, T_cnd_apply = 0,0,0,0,0 # 冷房大风量：3:00-6:00, 12:00-15:00
        
    print("Calculated energy:", P)
    P_totals_0701.append(P)
    Q_latents_0701.append(Q_latent_cal)

plot_onedayfitting(df_test_reheat_ftting, P_totals_0615, "upper left")
plot_onedayfitting(df_test_cooling_ftting, P_totals_0701, "upper center")

plot_onedayfitting_latent(df_test_reheat_ftting, Q_latents_0615, "upper left")
plot_onedayfitting_latent(df_test_cooling_ftting, Q_latents_0701, "upper center")

##################################
# two month 
##################################
def plot_parity(P_totals, y_test, color, y_upper):
    plt.figure(figsize=(15, 15))
    # 1. 绘制散点图
    plt.scatter(y_test, P_totals, color=color, alpha=0.8, s=100)
    # 2. 绘制 y = x 虚线，范围从 0 到 200
    reference_line = [0, y_upper]
    plt.plot(reference_line, reference_line, color='grey', linestyle='--', linewidth=2.2)
    # 3. 强制设置坐标轴范围为 0-200
    plt.xlim(0, y_upper)
    plt.ylim(0, y_upper)
    # 保持 1:1 的视觉比例，确保 y=x 是完美的 45 度角
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tick_params(axis='both', which='major', labelsize=14)
    # 图表修饰
    # plt.title('EC Prediction Accuracy')
    plt.xlabel('Measured', fontsize = 18)
    plt.ylabel('Predicted', fontsize = 18)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

test_data_XYplot = r"data\0614_0810_data_processed_0_7.csv"
df_test_XYplot = pd.read_csv(test_data_XYplot)
df_test_XYplot['Time'] = pd.to_datetime(df_test_XYplot['Time'])
df_test_reheat = df_test_XYplot[df_test_XYplot['mode_real'] == 'reheat'].copy().reset_index(drop=True)
df_test_cooling = df_test_XYplot[df_test_XYplot['mode_real'] == 'cooling'].copy().reset_index(drop=True)
y_test_reheat_P = df_test_reheat['Totalenergy']
y_test_cooling_P = df_test_cooling['Totalenergy']
y_test_reheat_latent = df_test_reheat['Q_latent']
y_test_cooling_latent = df_test_cooling['Q_latent']

P_totals_reheat = []   
P_totals_cooling = []
Q_latents_reheat = []
Q_latents_cooling = []

####################再热部分###################
for e in range(len(df_test_reheat)):
    print("-------------------------再热部分---------------------------")
    print("real energy",df_test_reheat["Totalenergy"][e])
    P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = ac_reheat.output(300, df_test_reheat["Xxiru"][e], df_test_reheat["Txiru"][e], df_test_reheat["Tout"][e], (df_test_reheat["Q_sensible"][e] + df_test_reheat["Q_latent"][e]), 10)
    print("Calculated energy:", P)
    P_totals_reheat.append(max(0,P))
    Q_latents_reheat.append(max(0,Q_latent_cal))

###################冷房部分####################
for e in range(len(df_test_cooling)):
    print("--------------------------冷房部分--------------------------")
    print("real energy",df_test_cooling["Totalenergy"][e])
    if df_test_cooling["volume_setting"][e] == "medium-low":
        print("冷房小风量")
        P, Lsensible_apply, Q_latent_cal, T_evp_apply, T_cnd_apply = ac_cool.ac_output(df_test_cooling["Q_total"][e],df_test_cooling["Txiru"][e],df_test_cooling["Xxiru"][e],df_test_cooling["Tout"][e], 576) # 冷房小风量：6:00-12:00, 15:00-21:00
    elif df_test_cooling["volume_setting"][e] == "medium-high":
        print("冷房中风量")
        P, Lsensible_apply, Q_latent_cal, T_evp_apply, T_cnd_apply = ac_cool.ac_output(df_test_cooling["Q_total"][e],df_test_cooling["Txiru"][e],df_test_cooling["Xxiru"][e],df_test_cooling["Tout"][e], 696) # 冷房中风量：6:00-12:00, 15:00-21:00
    elif df_test_cooling["volume_setting"][e] == "very-high":
        print("冷房大风量")
        P, Lsensible_apply, Q_latent_cal, T_evp_apply, T_cnd_apply = ac_cool.ac_output(df_test_cooling["Q_total"][e],df_test_cooling["Txiru"][e],df_test_cooling["Xxiru"][e],df_test_cooling["Tout"][e], 906) # 冷房大风量：3:00-6:00, 12:00-15:00
    else:
        P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = 0,0,0,0,0
    print("Calculated energy:", P)
    P_totals_cooling.append(P)
    Q_latents_cooling.append(Q_latent_cal)

plot_parity(P_totals_reheat, y_test_reheat_P, 'red', 400)
plot_parity(P_totals_cooling, y_test_cooling_P, 'red', 800)
plot_parity(Q_latents_reheat, y_test_reheat_latent, 'red', 1500)
plot_parity(Q_latents_cooling, y_test_cooling_latent, 'red', 1000)