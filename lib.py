import pandas as pd 
import numpy as np

###############################################
####### CONSTANT and CONVERSION CONSTAN #######
###############################################
KB 			= 1.3806488e-23 #boltzman constant
H  			= 6.62606957e-34 #plank constant
C  			= 2.99792458e10 #light speed
R  			= 8.3144621 #real gas constant
AV 			= 6.0221415e23 #Avogadro number
EhtokJmol  	= 2625.5002
AMUtoKG    	= 1.66053886e-27
CMtokJmol  	= 1.1962659192089765e-2 #C * H * AV change due to numerical errors
CMtoK      	= 1.4387862961655296
kjtokcal   	= 4.184
kJmoltoK   	= 120.2731159571
MHZtoK     	= 6.62606957 / 1.3806488 * 1e-5 # e-34 * 1e6 / e-23
AUconv     	= 1.66053886 * 1.3806488 / 6.62606957 / 6.62606957 * 1e18 # AMUtoKG * KB / H / H
UMAAAtoKGM 	= 1.660539e-47
###############################################
###############################################

###############################################
################## FUNCTIONS ##################
###############################################

def read_freq_gaussian(name_out):
	"""
	Read freq from g98.out file, and give back the freq in 1D np.array in cm-1
	name_out: g98.out file name/path
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

def read_freq_orca(name_out):
	"""
	Read freq from ORCA.hess file, and give back the freq in 1D np.array in cm-1
	name_out: ORCA.hess file name/path
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
	Read freq from .csv file, and give back the freq in 1D np.array in cm-1
	name_out: name.csv file name/path
	name_column: csv column name of the frequencies
	"""
	with open(name_out,"r") as data:
		df = pd.read_csv(data, delimiter = '\t', index_col=False)
	freq = pd.Series(df[name_column].to_numpy(dtype=float))
	freq = freq[~np.isnan(freq)]
	return np.array(freq)

def ZPE(freq):
	"""
	ZPE in kJ/mol from freq in cm-1, freq must be in np 1D array
	energy output is in kJ/mol
	"""
	return np.sum(freq * CMtokJmol / 2)

def kooji(t,alpha,beta,gamma):
	"""
	Arrehnius-kooji equation
	t: Temperature in K
	alpha, beta and gamma: the parameters in K
	"""
	return  alpha * (t / 300) ** beta * np.exp( - gamma / t)

def lamda_trasl(t,m):
	"""
	Molecules thermal wavelenght
	t: Temperature in K
	m: the mass of the absorbate molecole in AMU
	"""
	return H / np.sqrt(2 * np.pi * m * AMUtoKG * KB * t)

def q_trasl_2d(t,m,a):
	"""
	translational 2D partition function, for an ideal gas on a 2D surface
	t: Temperature in K
	m: the mass of the absorbate molecole in AMU
	a: the surface per adsorbed molecules
	"""
	return 2 * np.pi * m * AMUtoKG * KB * t * a / H / H

def q_rot(t,Rot_info,sim,rot_unit):
	"""
	rotational partition function
	t: Temperature in K
	Rot_info: rotational constant in cm-1 or inertial axis in amu * A**2
	sim: degeneracy in simmetry rotation
	rot_unit: if set to TRUE Rot_info must be the rotational constant in cm-1 (if in Mhz change CMtoK to MHZtoK), if set tu FALSE Rot_info must be the inertial axis in amu * A**2
	"""
	if rot_unit == True:
		return np.sqrt(np.pi) * np.sqrt(np.prod(np.array(Rot_info) * CMtoK * t)) / sim
	elif rot_unit == False:
		return np.sqrt(np.pi) / sim * np.power(8 * np.pi * np.pi * KB * t,1.5) * np.sqrt(np.prod(np.array(Rot_info) * UMAAAtoKGM )) / H / H / H

def q_vib(t,freq):
	"""
	vibrational partition function starting from level 1
	t: Temperature in K
	freq in cm-1
	"""
	return np.prod(1 / (1 - np.exp(- freq * CMtoK / t)))

def e_vib(t,freq,ZPE):
	"""
	vibrational internal energy with and without ZPE
	t: Temperature in K
	freq in cm-1
	ZPE: id set to True computed also the ZPE, if set to False not computed the ZPE
	energy output is in kJ/mol
	"""
	if ZPE == True:
		return np.sum(freq * CMtokJmol * (0.5 +  1 / (np.exp(freq * CMtoK / t) - 1))) 
	elif ZPE == False:
		return np.sum(freq * CMtokJmol / (np.exp(freq * CMtoK / t) - 1)) 

def BH_t(t,freq_m,freq_s,freq_c,be,be_zpe):
	"""
	BH(0) (i.e. BE with ZPE correction) corrected for thermal contribution
	t: Temperature in K
	freq in cm-1
	freq_m: desorbed molecule freq
	freq_s: surface after desorbed molecule freq
	freq_c: complex (mol+ surf) freq
	be: binding energy in kJ/mol
	be_zpe: if set to True the BE is not corrected for ZPE, if set to False is corrected for the ZPE (i.e. BH(0))
	energy output is in kJ/mol
	"""
	return be + e_vib(t,freq_m,be_zpe) + e_vib(t,freq_s,be_zpe) - e_vib(t,freq_c,be_zpe) + 4 * R * t * 1e-3

def pre_factor_HH(m,a,bh):
	"""
	prefactor Hebst-Hasegawa in s-1
	m: the mass of the absorbate molecole in AMU
	a: the surface per adsorbed molecules
	bh: binding energy in kJ/mol
	"""
	return np.sqrt(2 * bh * 1e3 / a / np.pi / np.pi / m / 1e-3)

def pre_factor_tait(t,m,a,sim,Rot_info,rot_unit):
	"""
	prefactor with Tait assumption in s-1
	t: Temperature in K
	m: the mass of the absorbate molecole in AMU
	a: the surface per adsorbed molecules
	sim: degeneracy in simmetry rotation
	Rot_info: inertial axis in amu * A**2
	rot_unit: if set to TRUE Rot_info must be the rotational constant in cm-1 (if in Mhz change CMtoK to MHZtoK), if set tu FALSE Rot_info must be the inertial axis in amu * A**2
	"""
	lmd 	= lamda_trasl(t,m)
	qrot 	= q_rot(t,Rot_info,sim,rot_unit)
	return KB * t / H * a / lmd / lmd * 1 * qrot

def pre_factor_harm(t,m,a,sim,Rot_info,rot_unit,freq_ts_mol,freq_ts_cluster,freq_ads):
	"""
	prefactor using vibrational partition function in harmoni approximation in s-1
	t: Temperature in K
	m: the mass of the absorbate molecole in AMU
	a: the surface per adsorbed molecules
	sim: degeneracy in simmetry rotation for the gas phase absorbate molecule
	Rot_info: inertial axis in amu * A**2
	rot_unit: if set to TRUE Rot_info must be the rotational constant in cm-1 (if in Mhz change CMtoK to MHZtoK), if set tu FALSE Rot_info must be the inertial axis in amu * A**2
	freq_ts_mol: freq in cm-1 for the gas phase absorbate molecule
	freq_ts_cluster: freq in cm-1 for cluster reference
	freq_ads: freq in cm-1 for the complex (adsorbed molecules + cluster)
	"""
	qtrl 			= q_trasl_2d(t,m,a)
	qrot 			= q_rot(t,Rot_info,sim,rot_unit)
	qvib_ts_mol 	= q_vib(t,freq_ts_mol)
	qvib_ts_cluster = q_vib(t,freq_ts_cluster)
	qvib_ts_ads 	= q_vib(t,freq_ads)
	return KB * t / H * qrot * qtrl * qvib_ts_mol * qvib_ts_cluster / qvib_ts_ads

def RH_DesRate(t,prefactor,be):
	"""
	Desorption rate in Red-Head approach
	t: Temperature in K
	prefactor in s-1
	be in kj/mol
	"""
	return  prefactor * np.exp( - be * kJmoltoK / t)

###############################################
###############################################
