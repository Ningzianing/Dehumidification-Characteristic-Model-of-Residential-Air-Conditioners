import reheat_dehumidification as rm
import cooling_dehumidification as cm

if __name__ == "__main__":
    experiment_data_cooling = r"data\0701data_cooling.csv"
    ac_cool = cm.cooling_aircon(500,2200,3300,115,425,960,906,2160, experiment_data_cooling, construct_model= True) 

    experiment_data = r"data\0615data_reheat.csv"
    ac_reheat = rm.reheat_aircon(500,2200,3300,115,425,960,906,2160,experiment_data, construct_model= True) 
    
    P, Lsensible_apply, Q_latent_cal, T_evp_apply, T_cnd_apply = ac_cool.ac_output(848.2075396,22.5294,0.010299202,22.556, 696)
    print(P, Q_latent_cal)

    P, T_evp_apply, T_cnd_apply, Q_latent_cal, SHF = ac_reheat.ac_output(300, 0.010561504, 22.8846, 23.23546667, 836.4658413, P_tolerance=10)
    print(P, Q_latent_cal)



