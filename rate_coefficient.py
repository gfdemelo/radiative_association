#!/usr/bin/env python


import numpy as np, scipy.constants as pc
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


""" Script para cálculo do rate coefficient de uma radiação associativa"""


def reduced_mass(mass1, mass2):
	return (mass1 * mass2)/	(mass1 + mass2) * pc.physical_constants['atomic unit of mass'][0]

T = np.linspace(300,13000, 100)

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
fit_transition = interp1d(transition_moment['r'], transition_moment['D'], kind='cubic')
potential = np.genfromtxt('esq', dtype=[('sigma', float),('Xpi', float)], skip_header=1, usecols=(1,3))
fit_potential = interp1d(transition_moment['r'], potential['Xpi'], kind='cubic')
fit_potential2 = interp1d(transition_moment['r'], potential['sigma'], kind='cubic')

r = np.linspace(3.1, 9, 100)

A_Einstein = 2.03e-6 * (((2*S + 1)*(2 - k_delta))/((2*S + 1)*(2 - k_delta))) * (0.1364)**2 * abs((min(fit_potential2(r)) - min(fit_potential(r))) * cm) **3 #(min(abs(potential['sigma'])) - min(abs(potential['Xpi'])) * cm)**3
print('Coeficiente de Einstein = %.4e' % (A_Einstein))
E = 1000 # em eV	#abs(potential['sigma'] - potential['Xpi']) * pc.physical_constants['Planck constant'][0]
integrand_r = lambda r: abs(A_Einstein) / ((1 - (fit_potential(r)*27.211)/E - b**2/r**2)**0.5)
f = quad(integrand_r,3.1,9)[0] 

cross_section = 4*np.pi*(reduced_mass(30.97376,31.97207)/(2*E))**0.5 * f * P  * quad(lambda b: b,0,np.inf)[0] 
print('Seção de choque =  %.4e' % (cross_section), sep='\n')


#Rate(T)
rates=[]
print('T',3*' ','-',3*' ','Rate')
for temperatura in T:
	coeff = (8 / (reduced_mass(30.97376,31.97207) * np.pi))**0.5 * (1 / (temperatura * pc.physical_constants['Boltzmann constant in eV/K'][0]))
	rate = coeff * quad(lambda e: e * cross_section * np.exp(-e/(pc.physical_constants['Boltzmann constant in eV/K'][0]*temperatura)), 0, np.inf)[0]
	#rate = coeff * quad(lambda e: e * np.sum(4*np.pi*(reduced_mass(30.97376,31.97207)/(2*e))**0.5 * (quad(lambda r: abs(A_Einstein) / ((1 - (fit_potential(r)*27.211)/e - b**2/r**2)**0.5),3.1,9)[0])  * P  * quad(lambda b: b,0,np.inf)[0]  )* np.exp(-e/(pc.physical_constants['Boltzmann constant in eV/K'][0]*temperatura)), 0, np.inf)[0]
	rates.append(rate)
	print('%d - %.14e' % (temperatura, rate))


#Plot
emin=min(fit_potential(r))
fig, axes = plt.subplots(nrows=2,ncols=2, figsize=(8,15))
ax1, ax2, ax3 = axes[0,0], axes[0,1], axes[1,0]#.get_gridspec()
axes[1,1].remove()
ax1.plot(r, (fit_potential(r) - emin)*cm)
ax1.plot(r, (fit_potential2(r) - emin)*cm)
ax2.plot(r, fit_transition(r), color='r')
ax2.set_title('Transition Moment')
ax1.set_title('CEPs PS')
ax3.plot(T, rates)
ax3.set_title('Rate')
ax3.set_xlabel('T / K')
ax3.set_ylabel('rate / cm$\mathrm{^3.s^{-1}}$')
plt.show()
