"""
Author: Sean Farrington
Mon Oct 30 17:08:38 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from scipy.optimize import minimize

def Apostolidis(Hct,cf):
    if cf<0.75:
        Hct_c = 0.3126*cf**2 - 0.468*cf + 0.1764
    else:
        Hct_c = 0.0012
        
    if Hct > Hct_c:
        tau_y = (Hct-Hct_c)**2 * (0.5084*cf+0.4517)**2 # Dyne/cm**2
    else:
        tau_y = 0
        
    eta_p = 1.67*10**-2
    T0 = 296.16
    T = 37 + 273.15    
    eta_c = eta_p*(1 + 2.0703*Hct + 3.7222*Hct**2)*np.exp(-7.0276*(1-T0/T)) # dyne.s/cm**2
    tau_y = tau_y*0.1 # Pa
    eta_c = eta_c*0.1 # Pa.s
    return tau_y,eta_c
    


def settling(ts,v_s):
    eta_p = 0.003 # Pa.s
    omega = 0.0105 # rad/s
    
    Rci = 27.722/2*10**-3 # m
    Rbi = 28.576/2*10**-3 # m
    Rbo = 32.997/2*10**-3 # m
    Rco = 33.989/2*10**-3 # m
    
    L = 40*10**-3 # m  NEED TO FIND TRUE LENGTH
    phi_o = 0.45 # volume fraction
    
    Ri_bar = (Rci+Rbi)/2
    Ro_bar = (Rbo+Rco)/2
    alpha = Ri_bar*Rbi**2/(Rbi-Rci) + Ro_bar*Rbo**2/(Rco-Rbo)
    
    gamma_i = omega*Ri_bar/(Rbi-Rci)
    gamma_o = omega*Ro_bar/(Rco-Rbo)
    shear_avg = (gamma_i + gamma_o)/2
    phis = phi_o*L/(L-v_s*ts)
    
    
    
    # Load in ML model for Casson parameterization
    filename1 = 'ML/finalized_model_yieldstress.sav'
    filename2 = 'ML/finalized_model_viscosity.sav'

    gp_yield = pickle.load(open(filename1,'rb'))
    gp_visc = pickle.load(open(filename2,'rb'))
    
    
    
    torque = []
    for i in range(len(ts)):
        # Apostolidis parameterization
        # tau_y,eta_c = Apostolidis(phis[i],0.3)
        
        # Farrington parameterization
        x_data = np.array([[phis[i],0.3]])
        tau_y = gp_yield.predict(x_data)/1000
        eta_c = gp_visc.predict(x_data)/1000
        
        L_p = v_s*ts[i]
        
        M_pT = 2*np.pi*L_p*omega*eta_p*alpha
        
        a = Rbi**2*(np.sqrt(tau_y)+np.sqrt(eta_c*gamma_i))**2 
        b = Rbo**2*(np.sqrt(tau_y)+np.sqrt(eta_c*gamma_o))**2
        M_sT = 2*np.pi*(L-L_p)*(a + b)
        
        torque = np.append(torque, M_pT + M_sT)
        
        # constant = 2*np.pi*omega*alpha
        # viscosity = (np.sqrt(tau_y/shear_avg) + np.sqrt(eta_c))**2
        # torque = np.append(torque,constant * (v_s*ts[i]*eta_p + (L-v_s*ts[i])*viscosity)) # N.m
        
    torque = torque*10**6 # muN.m
    return torque

###################################
ts = np.linspace(0,150,100)
v_s = 27.6/1000/3600 # m/s
###################################

torque = settling(ts,v_s)

file = 'ANALYSIS/SS_stepdowns.xlsx'
df = pd.read_excel(file,sheet_name='shear 0.35 1_s')


# def func(x):
#     x = x/1000/3600
#     file = 'ANALYSIS/SS_stepdowns.xlsx'
#     df = pd.read_excel(file,sheet_name='shear 0.35 1_s')
#     torque = settling(df['Step time'].to_numpy(),x)
#     torque_norm = torque/max(torque)
#     torque_exp = df['Torque']
#     torque_norm_exp = torque_exp/torque_exp.max()
#     torque_norm_exp = torque_norm_exp.to_numpy()
#     return np.sqrt(np.mean((torque_norm-torque_norm_exp)**2))

def func(x):
    x = x/1000/3600
    file = 'ANALYSIS/SS_stepdowns.xlsx'
    df = pd.read_excel(file,sheet_name='shear 0.35 1_s')
    torque = settling(df['Step time'].to_numpy(),x)
    torque_exp = df['Torque']
    torque_norm_exp = torque_exp.to_numpy()
    torque_norm = (torque - (max(torque) - max(torque_exp)))
    torque_norm = np.nan_to_num(torque_norm,nan = min(torque_norm))
    return np.sqrt(np.mean((torque_norm-torque_norm_exp)**2))
    
x0 = 1500
res = minimize(func, x0)

#%% Plotting
v_s_opt = res.x/1000/3600
torque_opt = settling(ts,v_s_opt)

fig,ax = plt.subplots(dpi=360)
ax.plot(ts,(torque_opt - (max(torque_opt)-df['Torque'].max())),'m--',linewidth=3,label='Fit Settling Velocity (2440 mm/hr)')
ax.plot(ts,(torque - (max(torque) - df['Torque'].max())),'b-',linewidth=3,label='Measured Settling Velocity (27.6 mm/hr)')
ax.plot(df['Step time'],df['Torque'],'gs',mec='k',mew='1',label='Rheology Experiment')
ax.set_ylabel('Torque, \u03BCN.m')
ax.set_xlabel('Time, s')
# ax.set_title('Shear Rate = 0.35 1/s')
plt.legend()
plt.savefig('FIGURES/sedimentation.png',dpi=600)
plt.show()

dataExport = pd.DataFrame({'Step time, s':df['Step time'],
                           'Torque, muN.m':df['Torque']})
theoryExport = pd.DataFrame({'Step time':ts,
                             'Torque, muN.m':torque})

# with pd.ExcelWriter('ANALYSIS/Required_SettlingData.xlsx') as writer:
#     dataExport.to_excel(writer,sheet_name='Data',index=False)
#     theoryExport.to_excel(writer,sheet_name='Theory',index=False)
    

