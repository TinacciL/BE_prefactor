import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from lib import BH_t, read_freq_orca, pre_factor_tait, pre_factor_harm, RH_DesRate, pre_factor_HH


### PARAMETERS and INITIALIZATION ###
step 			= 0.1	#Temperature integration step
beta 			= 0.01	#TPD Heating rate [K/s]
N_i  			= 1		#Number/density of species, this value do not affect the TPD since the y-axis can be rapresent as [a.u]
A	 			= 1e-19	#Surface per adsorbed molecules
t_in 			= 10	#Initial Temperature
t_fin			= 250	#Final Temperature
des_order		= 1		#Desorption order
T 				= np.arange(t_in, t_fin, step,dtype=float) 	#1D numpy array of Temperature values
TPD				= np.zeros(len(T),dtype=float) 				#1D numpy array of TPD values
Des_rate		= np.zeros(len(T),dtype=float) 				#1D numpy array of Desorption rate values

### SAMPLE INPUT INFORMATION ###
mol 			= read('data/CH3OH/ch3oh.xyz')	#Isolated adsorbated molecule structure path
mass	 		= mol.get_masses().sum()
inertia_moments = mol.get_moments_of_inertia()
rot_sim			= 1								#Rotational simmetry
BH_0			= 60.11							#BE corrected for ZPE, if not corrected used the BH_T function in the TPD loop

### READ FREQUENCIES ###
freq_isolated_mol 	= read_freq_orca('data/CH3OH/ch3oh.hess')
freq_isolated_surf 	= read_freq_orca('data/CH3OH/CH3OH_309/cluster_opt.hess')
freq_complex	 	= read_freq_orca('data/CH3OH/CH3OH_309/grain_mol_opt.hess')

tmp_N				= 0
for i in range(len(T)):
	bh_t			= BH_0 + BH_t(T[i],freq_isolated_mol,freq_isolated_surf,freq_complex,True) 	#Thermal correction to BE/BH(0)
	#Pre_factor		= pre_factor_HH(mass,A,bh_t) 												#prefactor with HH approximation
	#Pre_factor		= pre_factor_tait(T[i],mass,A,rot_sim,inertia_moments,False) 				#prefactor with Tait approximation
	Pre_factor		= pre_factor_harm(T[i],mass,A,rot_sim,inertia_moments,False,freq_isolated_mol,freq_isolated_surf,freq_complex) #prefactor with Harmonic freq
	Des_rate[i]		= RH_DesRate(T[i],Pre_factor,bh_t)
	if i == 0:
		tmp_N		= N_i - Des_rate[i] * N_i ** des_order / beta * step
		TPD[i]  	= Des_rate[i] * N_i ** des_order  / beta * step
	else:
		tmp 		= Des_rate[i] * tmp_N ** des_order / beta * step
		if tmp_N - tmp <= 0:
			tmp_N	= 0
		else:
			tmp_N	= tmp_N - tmp
			TPD[i]	= tmp

### SAVE TPD spectra and RATE CONSTANT in .csv file ####
df = pd.DataFrame()
df['T'] 			= T
df['TPD']			= TPD
df['Des_rate']		= Des_rate
with open("tmp/TPD.csv","w+") as output:
	output.write(df.to_csv(sep="\t", index=False))
