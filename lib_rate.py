import pandas as pd 
import numpy as np

KB = 1.3806488e-23
H  = 6.62606957e-34
C  = 2.99792458e10
R  = 8.3144621
AV = 6.0221415e23
EhtoKJMOL  = 2625.5002
AMUtoKG    = 1.66053886e-27
CMtokJmol  = 1.1962659192089765e-2 #C * H * AV change due to numerical errors
CMtoK      = 1.4387862961655296
kjtokcal   = 4.184
KJMOLtoK   = 120.2731159571
MHZtoK     = 6.62606957 / 1.3806488 * 1e-5 # e-34 * 1e6 / e-23
AUconv     = 1.66053886 * 1.3806488 / 6.62606957 / 6.62606957 * 1e18 # AMUtoKG * KB / H / H
UMAAAtoKGM = 1.660539e-47

def read_freq_gau(name_out):
	"""
	Read freq from g98.out file, and give back the freq in 1D np.array in cm-1
	"""
	freq = []
	with open(name_out,"r") as data:
		lines = data.readlines()
		for i,line in enumerate(lines):
			if ' Frequencies --' in line:
				tmp_f = line.split()
				tmp_f = tmp_f[2:]
				for i in range(len(tmp_f)):
					freq.append(float(tmp_f[i]))
	return np.array(freq)

def read_freq(name_out):
	"""
	Read freq from ORCA .hess file, and give back the freq in 1D np.array in cm-1
	"""
	freq = []
	with open(name_out,"r") as data:
		lines = data.readlines()
		for i,line in enumerate(lines):
			if '$vibrational_frequencies' in line:
				n_freq = int(lines[i+1].split()[0])
				for j in range(2+i+6,n_freq+i+2):
					tmp = lines[j].split()
					freq.append(float(tmp[1]))
	return np.array(freq)

def read_freq_from_csv(name_out,name_column):
	"""
	Read freq from csv file, and give back the freq in 1D np.array in cm-1
	name_out: file name .csv
	name_column: column name
	"""
	with open(name_out,"r") as data:
		df = pd.read_csv(data, delimiter = '\t', index_col=False)
	freq = pd.Series(df[name_column].to_numpy(dtype=float))
	freq = freq[~np.isnan(freq)]
	return np.array(freq)

def ZPE(freq):
	"""
	ZPE in kJmol from freq in cm-1, freq in np array
	energy output is in kJmol
	"""
	return np.sum(freq * CMtokJmol / 2)

def kooji(t,alpha,beta,gamma):
	"""
	Arrehnius-kooji equation
	t: Temperature
	alpha, beta and gamma: the parameters in K
	"""
	return  alpha * (t / 300) ** beta * np.exp( - gamma / t)

def Boltzmann_distr(T,states,degs):
	"""
	Boltzmann Distribution
	T: Temperature
	States: energy list/array of the states
	degs: degeneracy list/array of the states
	state_i: the state index (int) of which we whant to compute the population
	"""
	degs 	= np.array(degs)
	states 	= np.array(states)
	states  = states - states[-1]
	N 		= degs.sum() #number of states
	P   	= np.zeros(len(states),dtype=float)
	Z 		= 0 #partition function 
	for i in range(len(states)):
		Z = Z + degs[i] * np.exp(states[i] * KJMOLtoK / T)
	for i in range(len(states)):
		P[i] = degs[i] * np.exp(states[i] * KJMOLtoK / T) / N / Z 
	return(P)

def lamda_trasl(t,m):
	"""
	Molecules thermal wavelenght
	t: Temperature 
	m: the mass of the absorbate molecole
	"""
	return H / np.sqrt(2 * np.pi * m * AMUtoKG * KB * t)

def q_trasl_2d(t,m,a):
	"""
	translational 2D partition function, for a ideal gas on a 2D surface
	t: Temperature 
	m: the mass of the absorbate molecole
	a: the surface per adsorbed molecules
	"""
	return 2 * np.pi * m * AMUtoKG * KB * t * a / H / H
	#return 2 * np.pi * m * AUconv * t * a

def q_rot(t,Rc,sim):
	"""
	rotational partition function
	t: Temperature
	Rc: rotational constant in cm-1, if in Mhz change CMtoK to MHZtoK
	sim: degeneracy in simmetry rotation
	"""
	#return np.sqrt(np.pi) * np.power(8 * np.pi * np.pi * KB * t,1.5) * np.sqrt(np.prod(np.array(Rc) * CMtoK)) / sim / H / H
	return np.sqrt(np.pi) * np.sqrt(np.prod(np.array(Rc) * CMtoK * t)) / sim

def q_rot_inertia(t,Ic,sim):
	"""
	rotational partition function
	t: Temperature
	Ic: inertial axis in amu * A**2
	sim: degeneracy in simmetry rotation
	"""
	return np.sqrt(np.pi) / sim * np.power(8 * np.pi * np.pi * KB * t,1.5) * np.sqrt(np.prod(np.array(Ic) * UMAAAtoKGM )) / H / H / H

def q_vib(t,freq):
	"""
	vibrational partition function
	t: Temperature
	freq in cm-1
	"""
	return np.prod(1 / (1 - np.exp(- freq * CMtoK / t)))

def q_vib_lim_sum(t,freq,be):
	"""
	vibrational partition function summation over a give BE limit
	t: Temperature
	freq in cm-1
	N: quantum numbers state to sum up (not infinity)
	be: limit to summation to N state --> N = BE / freq
	"""
	tmp_f = np.zeros(len(freq),dtype=float)
	N = np.zeros(len(freq),dtype=float)
	#N = int(round(be/freq/CMtokJmol,0))
	N =np.round(be/freq/CMtokJmol,0).astype(int)
	N[N == 0] = 1
	for j in range(len(freq)):
		for i in range(0,N[j]):
			tmp_f[j] = tmp_f[j] + np.exp(- freq[j] * CMtoK * i / t)
	return np.prod(tmp_f)

def q_vib_morse(t,freq,be,N):
	"""
	vibrational partition function summation
	t: Temperature
	freq in cm-1
	be: dissocetion
	"""
	tmp_f = np.zeros(len(freq),dtype=float)
	for i in range(0,N):
		#tmp_f = tmp_f + np.exp(- freq * CMtoK * i / t) * np.exp( (freq * CMtoK) * (freq * CMtoK) / 4 / (be * KJMOLtoK) /t)
		tmp_f = tmp_f + np.exp(- freq * CMtoK / t + freq * CMtoK * freq * CMtoK * (i + 1) / 2 / (be * KJMOLtoK) / t)
	return np.prod(tmp_f)

def e_vib(t,freq):
	"""
	vibrational internal energy without ZPE
	t: Temperature
	freq in cm-1
	energy output is in kJmol
	"""
	#return np.sum(freq * CMtokJmol * (0.5 +  1 / (np.exp(freq * CMtoK / t) - 1))) #with ZPE
	return np.sum(freq * CMtokJmol / (np.exp(freq * CMtoK / t) - 1)) #without ZPE

def BH_t(t,freq_m,freq_s,freq_c,bh):
	"""
	BH(0) (i.e. BE with ZPE correction) corrected for thermal contribution
	t: Temperature
	freq in cm-1
	freq_m: desorbed molecule freq
	freq_s: surface after desorbed molecule freq
	freq_c: complex (mol+ surf) freq
	energy output is in kJmol
	"""
	return bh + e_vib(t,freq_m) + e_vib(t,freq_s) - e_vib(t,freq_c) + 4 * R * t * 1e-3 # 5/2 * R * t * 1e-3

def pre_factor_HH(m,a,bh):
	"""
	prefactor HH
	m: the mass of the absorbate molecole
	a: the surface per adsorbed molecules
	bh: binding energy in kJmol
	"""
	return np.sqrt(2 * bh * 1e3 / a / np.pi / np.pi / m / 1e-3)

def pre_factor_approx(t,m,a,sim,Ic):
	"""
	prefactor with Minnisale-Tait assumption
	t: Temperature
	m: the mass of the absorbate molecole
	a: the surface per adsorbed molecules
	sim: degeneracy in simmetry rotation
	Ic: inertial axis in amu * A**2
	"""
	lmd = lamda_trasl(t,m)
	qrot = q_rot_inertia(t,Ic,sim)
	return KB * t / H * a / lmd / lmd * 1 * qrot

def pre_factor_full(t,m,a,sim,Ic,freq_ts_mol,freq_ts_cluster,freq_ads):
	"""
	prefactor using vibrational partition function
	t: Temperature
	m: the mass of the absorbate molecole
	a: the surface per adsorbed molecules
	sim: degeneracy in simmetry rotation for the gas phase absorbate molecule
	Ic: rotational constant in cm-1 or Inerzia tensor in amu * A**2 for the gas phase absorbate molecule
	freq_ts_mol:freq in cm-1 for the gas phase absorbate molecule
	freq_ts_cluster:freq in cm-1 for cluster reference
	freq_ads:freq in cm-1 for the complex (adsorbed molecules + cluster)
	"""
	qtrl = q_trasl_2d(t,m,a)
	#qrot = q_rot(t,Ic,sim) #rotational constant
	qrot = q_rot_inertia(t,Ic,sim) #Inerzia tensor
	qvib_ts_mol = q_vib(t,freq_ts_mol)
	qvib_ts_cluster = q_vib(t,freq_ts_cluster)
	qvib_ts_ads = q_vib(t,freq_ads)
	return KB * t / H * qrot * qtrl * qvib_ts_mol * qvib_ts_cluster / qvib_ts_ads

def pre_factor_full_6n(t,m,a,sim,Ic,freq_ads):
	"""
	prefactor using vibrational partition function
	t: Temperature
	m: the mass of the absorbate molecole
	a: the surface per adsorbed molecules
	sim: degeneracy in simmetry rotation for the gas phase absorbate molecule
	Ic: inertial axis in amu * A**2
	freq_ts_mol:freq in cm-1 for the gas phase absorbate molecule
	freq_ts_cluster:freq in cm-1 for cluster reference
	freq_ads:freq in cm-1 for the complex (adsorbed molecules + cluster) ALL freq
	"""
	qtrl = q_trasl_2d(t,m,a)
	qrot = q_rot_inertia(t,Ic,sim)
	qvib_ts_ads = q_vib(t,freq_ads[:6])
	return KB * t / H * qrot * qtrl / qvib_ts_ads

def RH_DesRate(t,prefactor,be):
	"""
	Desorption rate in Red-Head approach
	t: Temperature
	prefactor in s-1
	be in kj/mol
	"""
	return  prefactor * np.exp( - be * KJMOLtoK / t)

def EdesFromTpeak_fix_par(t, pre):
	'''
	E_des in kJ/mol the formula is obtain considering prefactor and BE not dipendet on Temperature
	'''
	R = 8.3144621
	beta = 0.04
	return R * 1e-3 * t * (np.log( pre * t / beta) - 3.46)

def linear(x, a):
	"""
	linear function for fit: y = a * x
	"""
	return a * x

def linear_quote(x,a,b):
	"""
	linear function for fit: y = a * x + q
	"""
	return a * x + b

#from scipy import optimize
#def T_peak(t):
#	sample = str(370)
#	dH = 30.893808 #kJ/mol
#	Ic_h2o = [1.83,1.21,0.62] #amu * A**2
#	freq_h2o = np.array([1711.76,3737.17,3849.08]) #cm-1
#	mass_h20 = 18.02 #AMU
#	sim_water = 2
#	A_water = 1e-19 #surface area per adsorbed molecules cm2
#	path_0 = '/Users/tinaccil/Documents/code/ONIOM/ONIOM_H2O/' + sample + '/grain_mol_opt.hess'
#	path_1 = '/Users/tinaccil/Documents/code/ONIOM/ONIOM_H2O/' + sample + '/cluster_opt.hess'
#	freq_0 = read_freq(path_0)
#	freq_1 = read_freq(path_1)
#	beta = 0.04
#	qtrl = q_trasl_2d(t,mass_h20,A_water)
#	qrot = q_rot_inertia(t,Ic_h2o,sim_water) #Inerzia tensor
#	qvib_ts_mol = q_vib(t,freq_h2o)
#	qvib_ts_cluster = q_vib(t,freq_1)
#	qvib_ts_ads = q_vib(t,freq_0)
#	prefactor = KB * t / H * qrot * qtrl * qvib_ts_mol * qvib_ts_cluster / qvib_ts_ads
#	return dH * KJMOLtoK / t / t - np.exp(-dH * KJMOLtoK / t) / beta * prefactor
#sol = optimize.root(T_peak, 155,method='hybr')
