#!/usr/bin/env python


import numpy as np, scipy.constants as pc
from scipy.integrate import quad


""" Script para cálculo do rate coefficient de uma radiação associativa"""


def reduced_mass(mass1, mass2):
	return (mass1 * mass2)/	(mass1 + mass2) * pc.physical_constants['atomic mass unit-kilogram relationship'][0]

T = np.linspace(1,13000, 23)
coeff = (8 / (reduced_mass(15,16) * np.pi))**0.5 * (1 / T*pc.physical_constants['Boltzmann constant'][0])

#Parâmetro de impacto, raio P = raio S = 100 pm
b = 3.77945 		#bohr

#Probabilidade
S = 0.5 		#spin do estado molecular formado
S_a = 1.5 		# spin do P 
S_b = 1			# spin do S
L_a = 0			# momento angular do orbital do P
L_b = 1			# momento angular do orbital do S
k_delta = 0		#Kronecker delta, 1 para lambda=0 e 0 para os demais
P = ((2*S + 1)*(2 - k_delta))/((2*L_a + 1)*(2*S_a + 1)*(2*L_b + 1)*(2*S_b + 1))

#Seção de choque
cm = 219474.63
transition_moment = np.genfromtxt('Xpi-1sigma', dtype=[('r', float),('D',float)], comments='#', usecols=(0,1))
potential = np.genfromtxt('esq', dtype=[('sigma', float),('Xpi', float)], skip_header=1, usecols=(1,3))
A_Einstein = 2.03e-6 * ((2*S + 1)*(2 - k_delta))/((2*S + 1)*(2 - k_delta)) * abs(transition_moment['D'])**2 * (abs(potential['sigma'] - potential['Xpi']) * cm)**3

E = abs(potential['sigma'] - potential['Xpi']) * pc.physical_constants['Planck constant'][0]
integrand_r = A_Einstein / ((1 - potential['Xpi']/E - b**2/transition_moment['r']**2)**0.5)

cross_section = 4*np.pi*(reduced_mass(30.97376,31.97207)/2*E)**0.5 * np.sum(integrand_r) * np.sum(b) * P#quad(integrand_r,3.59,np.inf)[0] #* P   * quad(b,0,np.inf)[0] 
#print(cross_section)

#Rate(T)

rate = coeff * np.sum(E * cross_section * np.exp(-E/pc.physical_constants['Boltzmann constant'][0]*T))
print('T',3*' ','-',3*' ','Rate')
for i in range(len(rate)):
	print('%d - %.4e' % (T[i], rate[i]))
