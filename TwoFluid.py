"""
Author: Sean Farrington
Thu Aug 22 13:56:49 2024
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import os

def herschel_bulkley(shear_rate, tau_0, K, n):
    return tau_0 + K * shear_rate**n

def Casson(shear_rate,tau_C,eta_C):
    return np.square(np.sqrt(tau_C) + np.sqrt(eta_C*shear_rate))

def Cross(shear_rate,mu_inf,mu_0,K,n):
    return mu_inf + (mu_0 - mu_inf)/(1 + (K*shear_rate)**n)

def Cross_stress(shear_rate,mu_inf,mu_0,K,n):
    return shear_rate*mu_inf + shear_rate*(mu_0 - mu_inf)/(1 + (K*shear_rate)**n)

def FCross(shear_rate,mu_inf,mu_0,K,n,stress):
    stress_pred = Cross_stress(shear_rate,mu_inf,mu_0,K,n)
    
    return np.sqrt(np.mean((stress - stress_pred)**2)) 
    
def HB_viscosity(stress,tau_0, K, n):
    if stress < tau_0:
        mu_eff = 0.2
    else:
        mu_eff = stress*(K/(stress-tau_0))**(1/n)
    return mu_eff

def stretched_exp(t, a, b, c):
    return a * np.exp(-(t/b)) + c



my_path = os.path.dirname(__file__)
path = os.path.join(my_path, 'ANALYSIS/')

file = path + 'SS.xlsx'

df = pd.read_excel(file, header=0)

shear_rate_SS = df['ShearRate_1_s']
stress_SS = df['Stress_Pa']
visc_SS = df['Viscosity_mPa_s']/1000 # Steady state viscosity, Pa.s
cs = CubicSpline(stress_SS,visc_SS)
# Fit the model to the data
# popt, pcov = opt.curve_fit(herschel_bulkley, shear_rate_SS, stress_SS)
popt, pcov = opt.curve_fit(Cross,shear_rate_SS, visc_SS)

# Extract the fitted parameters
# tau_0, K, n = popt
mu_inf, mu_0, K, n = popt

fig,ax = plt.subplots()
# ax.plot(stress_SS, visc_SS,'ks')
ax.plot(shear_rate_SS,stress_SS,'ks')
xx = np.logspace(-2,3,1000)
ax.plot(xx, Cross_stress(xx,mu_inf,mu_0,K,n))
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()

file = path + 'SS_stepdowns.xlsx'
sheets = ['shear 0.1 1_s','shear 0.2 1_s','shear 0.35 1_s','shear 0.7 1_s',
          'shear 1.0 1_s','shear 2.0 1_s','shear 3.5 1_s','shear 7.0 1_s',
          'shear 10.0 1_s','shear 20.0 1_s','shear 35.0 1_s','shear 70.0 1_s',
          'shear 100.0 1_s','shear 200.0 1_s','shear 350.0 1_s','shear 700.0 1_s']

def fun(eps,visc_b,visc_w,shear_SS,stress_SS):
    return (stress_SS/visc_b) - (shear_SS - 2*eps*(stress_SS/visc_w))/(1-2*eps)

L = 500 # Gap width, um

#%% Cell-free layer Transience
file1 = path + 'DeltaTransience.xlsx'
with pd.ExcelWriter(file1) as writer:
    for i in range(6):
        deltas = np.array([])
        shear_bs = np.array([])
        shear_ws = np.array([])
        df = pd.read_excel(file,sheet_name=sheets[i], header=0)
        for stress in df['Stress'].to_numpy():
            bnds = ((0.0,None),)
            x0 = (0.01)
            sol = opt.minimize(FCross,x0,args=(mu_inf,mu_0,K,n,stress),bounds=bnds)
            shear_b = sol.x
            visc_b = stress/shear_b
            # visc_w = 0.001 # Viscosity of water (Pa.s)
            visc_w = 0.00275 # Viscosity of dextran medium (Pa.s)
            
            root = fsolve(fun,0,args=(visc_b,visc_w,shear_rate_SS[i],stress))
            
            delta = root*L
            deltas = np.append(deltas,delta)
           
            # shear_b = stress/visc_b
            shear_bs = np.append(shear_bs,shear_b)
            
            shear_w = stress/visc_w
            shear_ws = np.append(shear_ws,shear_w)
        
        df = pd.DataFrame({'Time_s':df["Step time"].to_numpy(),
                           'Strain':shear_rate_SS.iloc[i]*df["Step time"].to_numpy(),
                           'Deltas_um':deltas,
                           'Shear_b_invS':shear_bs,
                           'Shear_w_invS':shear_ws})
        
        df.to_excel(writer,sheet_name=sheets[i],index=False)
        fig,ax = plt.subplots()
        ax.plot(df["Time_s"],df['Deltas_um'])
        ax.set_ylabel('Delta')
        ax.set_xlabel('Time, s')

