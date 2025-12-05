#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 00:05:40 2025

@author: andr3s
"""
#%%
import os 
import pickle
import numpy as np
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from freegsnke import build_machine
from freegsnke.jtor_update import ConstrainPaxisIp
from freegsnke.jtor_update import ConstrainBetapIp #para usar el perfil con Beta
from freegsnke import equilibrium_update
from freegsnke import GSstaticsolver
from freegsnke.inverse import Inverse_optimizer
#%%

tokamak = build_machine.tokamak(
    passive_coils_path = "/home/andr3s/Documentos/SS/Códigos/FreeGSNKE/TCV/machine_configs/TCV/passive_coils.pickle",
    active_coils_path="/home/andr3s/Documentos/SS/Códigos/FreeGSNKE/TCV/machine_configs/TCV/active_coils4.pickle",
    limiter_path="/home/andr3s/Documentos/SS/Códigos/FreeGSNKE/TCV/machine_configs/TCV/limiter.pickle",
    wall_path="/home/andr3s/Documentos/SS/Códigos/FreeGSNKE/TCV/machine_configs/TCV/wall.pickle",
    magnetic_probe_path="/home/andr3s/Documentos/SS/Códigos/FreeGSNKE/TCV/machine_configs/TCV/magnetic_probes.pickle",
)

#%%
eq = equilibrium_update.Equilibrium(
    tokamak=tokamak,      # provide tokamak object
    Rmin=0.1, Rmax=1.8,   # radial range
    Zmin=-1.3, Zmax=1.3,  # vertical range
    nx=65,                # number of grid points in the radial direction (needs to be of the form (2**n + 1) with n being an integer)
    ny=129,               # number of grid points in the vertical direction (needs to be of the form (2**n + 1) with n being an integer)
    # psi=plasma_psi
)
# Parámetros físicos dinámicos
PAXIS    = float(os.environ.get("PRESION_PAXIS", 40e3))       # presión en el eje magnético [Pa]
IP       = float(os.environ.get("CORRIENTE_IP", 1e6))        # corriente total del plasma [A]
Bt       = float(os.environ.get("CAMPO_BT", 1.8))                  # Campo magnético toroidal B_tor [T]
Ra       = 0.88                                            # Radio mayor del tokamak
FVAC     = Ra*Bt                                           # fvac = R * B_tor 
ALPHA_M  = float(os.environ.get("PARAM_AM", 2))            # parámetro del perfil de presión
ALPHA_N  = float(os.environ.get("PARAM_AN", 2))            # parámetro del perfil de corriente      
# BETAP = 0.125
SOLENOID_CURRENT = 30e3 #ahora se puede ajustar la corriente como una variable
N_rho = 200 

#perfil para usarse con I
profiles = ConstrainPaxisIp(
    eq=eq,        # equilibrium object
    paxis=PAXIS,    # profile object
    Ip=IP,       # plasma current
    fvac=FVAC,     # fvac = rB_{tor}
    alpha_m=ALPHA_M,  # profile function parameter
    alpha_n=ALPHA_N   # profile function parameter
)
'''
#Perfil para usarse con beta
profiles_beta = ConstrainBetapIp(
    eq=eq,
    betap=BETAP,
    Ip=IP,
    fvac=FVAC,
    alpha_m=ALPHA_M,
    alpha_n=ALPHA_N
)
'''
#%%

# Ruta base controlada por variable de entorno
path_resultados = os.environ.get("RESULTADOS_PATH", "./Resultados_TCVSIRM/Default")
if not os.path.exists(path_resultados):
    os.makedirs(path_resultados)  # Asegura que la carpeta existe

# Nombre del entorno o configuración
N = "Triang_Pos"  
nombre_base = "TCV_SI_" + N

# Timestamp para evitar sobrescritura, le agrega datos de tiempo a los resultados obtenidos
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Nombres de archivos con fecha
nombre_cons = f"{nombre_base}.png"
nombre_img = f"{nombre_base}_{timestamp}.png"
nombre_csv = f"{nombre_base}_{timestamp}.csv"
nombre_pickle = f"{nombre_base}_{timestamp}.pk"

# Rutas completas para guardar
path_tablas = path_resultados
path_imagenes = path_resultados
path_pickle = path_resultados

#%%
Rx = 0.6499      # X-point radius
Zx = -0.2393      # X-point height
Rout = 1.0806    # outboard midplane radius
Rin = 0.6266    # inboard midplane radius

# set desired null_points locations
# this can include X-point and O-point locations
null_points = [[Rx], [Zx]]

# set desired isoflux constraints with format 
# isoflux_set = [isoflux_0, isoflux_1 ... ] 
# with each isoflux_i = [R_coords, Z_coords]
isoflux_set = np.array([[[Rx,0.88, Rin, Rout, 0.9174, 1.0786, 0.6474, 0.88], 
                         [Zx, 0.4607, 0.,0., -0.1868, 0.2777, 0.2777, -0.2091]]])
           
# instantiate the freegsnke constrain object

constrain = Inverse_optimizer(null_points=null_points,
                              isoflux_set=isoflux_set)

##################

eq.tokamak.set_coil_current('Solenoid', SOLENOID_CURRENT)
eq.tokamak['Solenoid'].control = True  # asegura que la corriente de esta bobina queda fija
#el solenoide debe quedar fijo porque sirve para generar un campo e inducir corriente al plasma
#ajustar en true, permitirá que el solucionador ajuste la corriente
#%%
#solucionador
# Definir tolerancias como variables
tol_residuo = 6e-3
tol_psit = 1e-3
GSStaticSolver = GSstaticsolver.NKGSsolver(eq) 
GSStaticSolver.solve(eq=eq, 
                     profiles=profiles, 
                     constrain=constrain, 
                     target_relative_tolerance=tol_residuo, 
                     target_relative_psit_update=tol_psit,
                     verbose=True,
                     l2_reg=np.array([1e-12]*25 + [1e-6]),  
                     )
#%%

# Construcción del perfil
# Precalcular valores que se usan más de una vez
AR = eq.aspectRatio()
Rmag, Zmag, psi_mag = eq.magneticAxis()
Rgeo, Zgeo = eq.geometricAxis()
S1, S2 = eq.innerOuterSeparatrix()

# Construir perfil
perfil_resumen = [
    ["PERFIL DE ARRANQUE", '', ''],
    ["Corriente de entrada", IP, 'Amps'],
    ["Radio mayor", Ra, 'm'],
    ["Presión", PAXIS, 'Pa'],
    ["Campo toroidal Bt", Bt, 'T'],
    [r'Vacío $F = RB_t$', FVAC, ''],
    ['FORMA DEL PLASMA', '', ''],
    ["Beta poloidal", eq.poloidalBeta1, ''],
    ["Beta toroidal", eq.toroidalBeta1, ''],
    ["Beta total normalizada", eq.normalised_total_Beta(), ''],
    ["Elongación geométrica", eq.geometricElongation(), ''],
    ["Triangularidad", eq.triangularity(), ''],
    ["Interseca la pared", eq.intersectsWall(), ''], #true el plasma toca la pared, false no toca 
    ["Aspect ratio", AR, ''],
    ["Aspect ratio inverso", 1.0 / AR if AR != 0 else 'N/A', ''],
    ["Radio menor", eq.minorRadius(), 'm'],
    ["Volumen del plasma", eq.plasmaVolume(), 'm³'],
    ['', '', ''],
    ['PARÁMETROS DEL PLASMA', '', ''],
    ["Corriente total del plasma", eq.plasmaCurrent(), 'Amps'],
    ["Presión en el eje", eq.pressure(np.array([0.0]))[0], 'Pa'],
    ["Presión promedio", np.mean(eq.pressure(eq.rho_1D(N_rho))), 'Pa'],
    ["Energía térmica", "N/A", 'J'],
    ["Parámetro am", ALPHA_M, ''],
    ["Parámetro an", ALPHA_N, ''],
    ["Lambda", profiles.L, ''],
    ["Beta0", profiles.Beta0, ''],
    ['', '', ''],
    ["Eje geométrico", f"({Rgeo:.3f}, {Zgeo:.3f})", '(R,Z)'],
    ["Eje magnético", f"({Rmag:.3f}, {Zmag:.3f}, {psi_mag:.3e})", '(R,Z,ψ)'],
    ["Separatriz en Z=0", f"({S1:.3f}, {S2:.3f})", '(R_in, R_out)'],
    ["Desplazamiento de Shafranov", eq.shafranov_shift(), '(ΔR, ΔZ)'],
    ['', '', ''],
    ['COORDENADAS DE CONTROL', '', ''],
    ["Punto X (Rx,Zx)", [Rx, Zx], 'm'],
    ["Puntos nulos (X/O)", null_points, '(R,Z)'],
    ["Restricciones isoflux", isoflux_set.tolist(), '(R,Z)'],
    ['', '', ''],
    ['TOLERANCIAS DEL SOLUCIONADOR', '', ''],
    ["Tolerancia objetivo (residuo)", tol_residuo, ''],
    ["Tolerancia objetivo (actualización de psit)", tol_psit, ''],
    ['', '', '']
]

# Encabezado
encabezado = ['Parámetro', 'Valor', 'Unidad']

# DataFrame
df_perfil = pd.DataFrame(perfil_resumen, columns=encabezado)

# Guardar en CSV
df_perfil.to_csv(path_tablas + '/Perfil_' + nombre_csv, index=False)

#%%

# Construcción del tokamak
fig1, ax1 = plt.subplots(1, 1, figsize=(4, 8), dpi=80)
plt.tight_layout()

tokamak.plot(axis=ax1, show=False)
ax1.plot(tokamak.limiter.R, tokamak.limiter.Z, color='k', linewidth=1.2, linestyle="--", label='Limiter')
ax1.plot(tokamak.wall.R, tokamak.wall.Z, color='k', linewidth=1.2, linestyle="-", label='Wall')

ax1.grid(alpha=0.5)
ax1.set_aspect('equal')
ax1.set_xlim(0.3, 1.85)
ax1.set_ylim(-1.3, 1.3)
ax1.set_xlabel(r'Major radius, $R$ [m]')
ax1.set_ylabel(r'Height, $Z$ [m]')

fig1.savefig(path_imagenes+'/cons_'+ nombre_cons, format='png', dpi=300, bbox_inches='tight')

#%%
#Factor de seguridad
# Evitar extremos numéricos
eps = 0.01
psiN = np.linspace(eps, 1 - eps, N_rho)

# Asegurar que el interpolador esté sincronizado
eq._updatePlasmaPsi(eq.plasma_psi)

# Calcular el factor de seguridad
q = eq.q(psiN)

# Guardar CSV
nombre_csv_completo = path_resultados + '/q_' + nombre_csv
with open(nombre_csv_completo, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['# Perfil del factor de seguridad q(ψ_N)'])
    writer.writerow(['# N_rho =', N_rho])
    writer.writerow(['psiN', 'q(psiN)'])
    for psi, q_val in zip(psiN, q):
        writer.writerow([f"{psi:.6f}", f"{q_val:.6f}"])

# Graficar
plt.figure(figsize=(7, 5), dpi=120)
plt.plot(psiN, q, label=r'$q(\psi_N)$', color='darkblue', linewidth=2)
plt.xlabel(r'$\psi_N$ (flujo normalizado)')
plt.ylabel(r'$q$ (factor de seguridad)')
plt.title('Perfil del factor de seguridad $q(\psi_N)$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(path_imagenes + '/q_' + nombre_img, format='png', dpi=300)
plt.show()

#%%

#equilibrio sin constricciones
fig_eq1, ax_eq1 = plt.subplots(figsize=(6, 8), dpi=100)

ax_eq1.grid(zorder=0, alpha=0.75)
ax_eq1.set_aspect('equal')
eq.tokamak.plot(axis=ax_eq1, show=False)
ax_eq1.fill(tokamak.wall.R, tokamak.wall.Z, color='k', linewidth=1.2, facecolor='w', zorder=0)
eq.plot(axis=ax_eq1, show=False)
ax_eq1.set_xlim(0.3, 1.85)
ax_eq1.set_ylim(-1.3, 1.3)

# Guardado
fig_eq1.savefig(path_imagenes + '/EqS_' + nombre_img, format='png', dpi=300, bbox_inches='tight')

#equilibrio con constricciones
fig_eq2, ax_eq2 = plt.subplots(figsize=(6, 8), dpi=100)

ax_eq2.grid(zorder=0, alpha=0.75)
ax_eq2.set_aspect('equal')
eq.tokamak.plot(axis=ax_eq2, show=False)
ax_eq2.fill(tokamak.wall.R, tokamak.wall.Z, color='k', linewidth=1.2, facecolor='w', zorder=0)
eq.plot(axis=ax_eq2, show=False)
constrain.plot(axis=ax_eq2, show=True)
ax_eq2.set_xlim(0, 2.15)
ax_eq2.set_ylim(-1.3, 1.3)

# Guardado
fig_eq2.savefig(path_imagenes + '/EqC_' + nombre_img, format='png', dpi=300, bbox_inches='tight')

#%%
#separatrix
# Obtener coordenadas de la separatriz
RZ = eq.separatrix()
R = RZ[:, 0]
Z = RZ[:, 1]

# Límites visuales dinámicos
Rmin, Rmax = np.min(R), np.max(R)
Zmin, Zmax = np.min(Z), np.max(Z)

# Crear figura
plt.figure('Separatrix', figsize=(6, 5), dpi=120)
plt.title('Plasma Separatrix')
plt.plot(R, Z, label='Triangularidad: %.3f' % eq.triangularity(), color='crimson', linewidth=2)
plt.xlim(Rmin - 0.1, Rmax + 0.1)
plt.ylim(Zmin - 0.1, Zmax + 0.1)
plt.xlabel(r'$R$ [m]')
plt.ylabel(r'$Z$ [m]')
plt.legend(loc="upper right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Guardar figura con nombre sincronizado
plt.savefig(path_imagenes + '/Separatrix_' + nombre_img, format='png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#%%
# Obtener presión sobre la malla (usando psi normalizado manualmente)
psiN_malla = eq.psiNRZ(eq.R, eq.Z)
pres_malla = profiles.pressure(psiN_malla)

# Obtener densidad de corriente toroidal sobre la malla
jtor_malla = profiles.Jtor(eq.R, eq.Z, eq.psi(), eq.psi_bndry)

# Corte central en Z
idx_Z = pres_malla.shape[1] // 2
R_corte = eq.R[:, idx_Z]
pres_corte = pres_malla[:, idx_Z]
jtor_corte = jtor_malla[:, idx_Z]

# Posición del eje magnético
Ax = eq.Rmagnetic()

# Crear figura doble
plt.figure('Presión y Corriente vs R', figsize=(13, 6), dpi=120)

# Subplot: presión
plt.subplot(1, 2, 1)
plt.plot(R_corte, pres_corte, label='Presión', color='royalblue')
plt.axvline(Ax, linestyle=':', color='gray', label='Eje magnético')
plt.title('Presión en corte central')
plt.xlabel(r'$R$ [m]')
plt.ylabel(r'$P$ [Pa]')
plt.grid(True, linestyle=':')
plt.legend()

# Subplot: corriente toroidal
plt.subplot(1, 2, 2)
plt.plot(R_corte, jtor_corte, label='J toroidal', color='darkred')
plt.axvline(Ax, linestyle=':', color='gray', label='Eje magnético')
plt.title('Densidad de corriente toroidal en corte central')
plt.xlabel(r'$R$ [m]')
plt.ylabel(r'$J$ [A/m²]')
plt.grid(True, linestyle=':')
plt.legend()

# Guardar figura
plt.tight_layout()
plt.savefig(path_imagenes + '/P_J_' + nombre_img, format='png', dpi=300)
plt.show()
plt.close()
#%%
#campos totales Tokamak
# Obtener el objeto Tokamak
Tokamak = eq.getMachine()

# Separatrices y eje magnético
S1, S2 = eq.innerOuterSeparatrix()
Ax = eq.magneticAxis()[0]

# Corte en Z = 0
idx_Z = np.argmin(np.abs(eq.Z[0, :] - 0))
R_corte = eq.R[:, idx_Z]
Z_corte = np.full_like(R_corte, 0.0)

# Campos del tokamak
Br_tokamak = Tokamak.Br(R_corte, Z_corte) #tokamak
Bz_tokamak = Tokamak.Bz(R_corte, Z_corte) #tokamak
Br_plasma = eq.plasmaBr(R_corte, Z_corte) #plasma
Bz_plasma = eq.plasmaBz(R_corte, Z_corte) #plasma
Bpol_total  = eq.Bpol(R_corte, Z_corte)  # incluye plasma + tokamak
Btor_total = eq.Btor(R_corte, Z_corte)  # incluye plasma + tokamak

# Figura
plt.figure('B_Tokamak_Plasma', figsize=(12, 8), dpi=120)

# Br del tokamak
plt.subplot(2, 2, 1)
plt.plot(R_corte, Br_tokamak, label='Br Tokamak')
plt.axvline(Ax, ls=':', label='Eje magnético')
plt.axvline(S1, ls='-.', c='red', label='Separatrix')
plt.axvline(S2, ls='-.', c='red')
plt.title(r'Campo Magnético $B_r$ Tokamak')
plt.xlabel('Radio Menor (R)')
plt.ylabel(r'$B_r$ [Teslas]')
plt.grid()
plt.legend()

# Bz del tokamak
plt.subplot(2, 2, 3)
plt.plot(R_corte, Bz_tokamak, label='Bz Tokamak')
plt.axvline(Ax, ls=':')
plt.axvline(S1, ls='-.', c='red')
plt.axvline(S2, ls='-.', c='red')
plt.title(r'Campo Magnético $B_z$ Tokamak')
plt.xlabel('Radio Menor (R)')
plt.ylabel(r'$B_z$ [Teslas]')
plt.grid()
plt.legend()

# B total
plt.subplot(2, 2, 2)
plt.plot(R_corte, Btor_total, label=r'$B_\phi$ Total')
plt.axvline(Ax, linestyle=':', label='Eje magnético')
plt.axvline(S1, ls='-.', c='red')
plt.axvline(S2, ls='-.', c='red')
plt.title('Campo Magnético Toroidal Total')
plt.xlabel('Radio Menor (R)')
plt.ylabel(r'$B_\phi$ [Teslas]')
plt.grid()
plt.legend()

# Título general de la figura
plt.suptitle("Campos magnéticos tokamk", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(path_imagenes + '/Btok' + nombre_img, format='png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
#%%
#Campos totales plasma
# Figura
plt.figure('B_p, B_t y B_T', figsize=(12, 8), dpi=120)

# Br del plasma
plt.subplot(2, 2, 1)
plt.plot(R_corte, Br_plasma, label='Br plasma')
plt.axvline(Ax, linestyle=':', label='Eje magnético')
plt.axvline(S1, ls='-.', c='red', label='Separatrix')
plt.axvline(S2, ls='-.', c='red')
plt.title(r'Campo Magnético $B_r$ Plasma')
plt.xlabel(r'$R$ [m]')
plt.ylabel(r'$B_r$ [T]')
plt.grid()
plt.legend()

# Bz del plasma
plt.subplot(2, 2, 3)
plt.plot(R_corte, Bz_plasma, label='Bz plasma')
plt.axvline(Ax, linestyle=':')
plt.axvline(S1, ls='-.', c='red')
plt.axvline(S2, ls='-.', c='red')
plt.title(r'Campo Magnético $B_z$ Plasma')
plt.xlabel(r'$R$ [m]')
plt.ylabel(r'$B_z$ [T]')
plt.grid()
plt.legend()

# B poloidal total
plt.subplot(2, 2, 2)
plt.plot(R_corte, Bpol_total, label='B poloidal')
plt.axvline(Ax, linestyle=':')
plt.axvline(S1, ls='-.', c='red')
plt.axvline(S2, ls='-.', c='red')
plt.title('Campo Magnético Poloidal')
plt.xlabel(r'$R$ [m]')
plt.ylabel(r'$B_\theta$ [T]')
plt.grid()
plt.legend()

# B toroidal total
plt.subplot(2, 2, 4)
plt.plot(R_corte, Btor_total, label='B toroidal')
plt.axvline(Ax, linestyle=':')
plt.axvline(S1, ls='-.', c='red')
plt.axvline(S2, ls='-.', c='red')
plt.title('Campo Magnético Toroidal')
plt.xlabel(r'$R$ [m]')
plt.ylabel(r'$B_\phi$ [T]')
plt.grid()
plt.legend()

# Título general de la figura
plt.suptitle("Campos magnéticos plasma", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(path_imagenes + '/Bplas_' + nombre_img, format='png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
#%%

# Obtener las corrientes del equilibrio
inverse_current_values = eq.tokamak.getCurrents()

# Guardar en pickle
with open(path_pickle + '/Coil_currents_' + nombre_pickle, 'wb') as f:
    pickle.dump(inverse_current_values, f)
print(f"Corrientes guardadas en Pickle: {path_pickle}")

with open(path_tablas + '/Coil_currents_' + nombre_csv, mode='w', newline='') as file:
    writer = csv.writer(file)

    if isinstance(inverse_current_values, dict):
        writer.writerow(["Coil", "Current [A]"])
        for coil, current in inverse_current_values.items():
            writer.writerow([coil, f"{current:.2f}"])
    elif hasattr(inverse_current_values, "__iter__"):
        writer.writerow(["Index", "Current [A]"])
        for i, current in enumerate(inverse_current_values):
            writer.writerow([f"Coil_{i}", f"{current:.2f}"])
    else:
        raise TypeError("Formato de corriente no reconocido")