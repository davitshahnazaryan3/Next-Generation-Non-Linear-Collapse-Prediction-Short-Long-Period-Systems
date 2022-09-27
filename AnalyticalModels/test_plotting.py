import numpy as np
import matplotlib.pyplot as plt

# Load some scripts from Github
import sys

from AnalyticalModels.vidic import Vidic_et_al_1994
from AnalyticalModels.newmark_hall import Newmark_Hall_1992
from AnalyticalModels.krawinkler_nassar import Krawinkler_Nassar_1992
from AnalyticalModels.miranda import Miranda_1993

# Set the values to plot for
mu_list = np.array([4])
T_list = np.linspace(0.01, 10, 100)

# Initialise arrays
Vea94 = np.zeros((np.size(mu_list), np.size(T_list)))
NH92 = np.zeros((np.size(mu_list), np.size(T_list)))
KN92 = np.zeros((np.size(mu_list), np.size(T_list)))
M93 = np.zeros((np.size(mu_list), np.size(T_list)))

Tc = 0.5
Tcc = .4
ah = 2
site = 'rock'
Tg = 1.

# Get the values
for mu in mu_list:
    mu_ind = np.where(mu == mu_list)[0]
    for T in T_list:
        T_ind = np.where(T == T_list)[0]
        Vea94[mu_ind, T_ind] = Vidic_et_al_1994(mu,T,Tc)
        NH92[mu_ind, T_ind] = Newmark_Hall_1992(mu,T,Tcc, Tc)
        KN92[mu_ind, T_ind] = Krawinkler_Nassar_1992(mu,T,ah)
        M93[mu_ind, T_ind] = Miranda_1993(mu,T,site,Tg)

ltype = ['-', '--', ':']
plt.figure(figsize=(1.5*2.36,2.36))
plt.rcParams.update({'font.size': 10})
for mu in mu_list:
    mu_ind = np.where(mu == mu_list)[0][0]
    if mu_ind == 0:
        plt.plot(T_list, Vea94[mu_ind, :], color = 'blue', label='Vidic et al. (1994)', linestyle = ltype[mu_ind])
        plt.plot(T_list, NH92[mu_ind, :], color = 'red', label='Newmark & Hall (1992)', linestyle = ltype[mu_ind])
        plt.plot(T_list, KN92[mu_ind, :], color = 'green', label='Krawinkler & Nassar (1992)', linestyle = ltype[mu_ind])
        plt.plot(T_list, M93[mu_ind, :], color = 'orange', label='Miranda (1993)', linestyle = ltype[mu_ind])
    else:
        plt.plot(T_list, Vea94[mu_ind, :], color = 'blue', linestyle = ltype[mu_ind])
        plt.plot(T_list, NH92[mu_ind, :], color = 'red', linestyle = ltype[mu_ind])
        plt.plot(T_list, KN92[mu_ind, :], color = 'green', linestyle = ltype[mu_ind])
        plt.plot(T_list, M93[mu_ind, :], color = 'orange', linestyle = ltype[mu_ind])
plt.ylabel(r'Force Reduction Factor, $R$')
plt.xlabel(r'Period, $T$ [s]')
plt.xscale('log')
plt.legend(fontsize=6, loc=0)
plt.grid()
plt.show()
# plt.savefig('Test-Plotting-R-T.pdf', bbox_inches='tight')
plt.close()


# The previous plots were al in terms of R-T for a given mu, lets do a R_mu for a given T

mu_list = np.linspace(1, 8, 100)
T_list = np.array([0.5])

# Initialise arrays
Vea94 = np.zeros((np.size(mu_list), np.size(T_list)))
NH92 = np.zeros((np.size(mu_list), np.size(T_list)))
KN92 = np.zeros((np.size(mu_list), np.size(T_list)))
M93 = np.zeros((np.size(mu_list), np.size(T_list)))

# Get the values
for mu in mu_list:
    mu_ind = np.where(mu == mu_list)[0]
    for T in T_list:
        T_ind = np.where(T == T_list)[0]
        Vea94[mu_ind, T_ind] = Vidic_et_al_1994(mu,T,Tc)
        NH92[mu_ind, T_ind] = Newmark_Hall_1992(mu,T,Tcc, Tc)
        KN92[mu_ind, T_ind] = Krawinkler_Nassar_1992(mu,T,ah)
        M93[mu_ind, T_ind] = Miranda_1993(mu,T,site,Tg)

ltype = ['-', '--', ':']
plt.figure(figsize=(1.5*2.36,2.36))
plt.rcParams.update({'font.size': 10})
for T in T_list:
    T_ind = np.where(T == T_list)[0][0]
    if T_ind == 0:
        plt.plot(mu_list, Vea94[:, T_ind], color = 'blue', label='Vidic et al. (1994)', linestyle = ltype[T_ind])
        plt.plot(mu_list, NH92[:, T_ind], color = 'red', label='Newmark & Hall (1992)', linestyle = ltype[T_ind])
        plt.plot(mu_list, KN92[:, T_ind], color = 'green', label='Krawinkler & Nassar (1992)', linestyle = ltype[T_ind])
        plt.plot(mu_list, M93[:, T_ind], color = 'orange', label='Miranda (1993)', linestyle = ltype[T_ind])
    else:
        plt.plot(mu_list, Vea94[:, T_ind], color = 'blue', linestyle = ltype[T_ind])
        plt.plot(mu_list, NH92[:, T_ind], color = 'red', linestyle = ltype[T_ind])
        plt.plot(mu_list, KN92[:, T_ind], color = 'green', linestyle = ltype[T_ind])
        plt.plot(mu_list, M93[:, T_ind], color = 'orange', linestyle = ltype[T_ind])
plt.ylabel(r'Force Reduction Factor, $R$')
plt.xlabel(r'Ductility, $\mu$')
plt.legend(fontsize=6, loc=0)
plt.grid()
plt.show()
# plt.savefig('Test-Plotting-R-mu.pdf', bbox_inches='tight')
plt.close()
