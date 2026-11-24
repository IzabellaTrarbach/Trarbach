#!/usr/bin/env python3
"""Simulador de malha fechada do circuito RC (Vc3/Vin) para comparar com a bancada.

Uso:
  python3 simula_bancada.py autoteste
  python3 simula_bancada.py comparar arquivo.csv --kp 3.523 --ti 90.99 --td 11.114 [--alpha 0.64] [--modo I_PD]
                                                  [--sat] [--aw clamp] [--rp 133600 --rp-t 50]
Tudo em % da escala (100 % = 5 V) e segundos. Planta: FT completa (modelo fisico, sem reducao)."""
import sys, argparse, numpy as np
from scipy.linalg import expm

R = np.array([22000, 46800, 32600, 66800.])      # ohm  (EDITE com os SEUS valores medidos)
C = np.array([947e-6, 947e-6, 955e-6])           # F
g = 1 / R

def A_mat(gp=0.0):
    return np.array([[-(g[0]+g[1])/C[0], g[1]/C[0], 0],
                     [g[1]/C[1], -(g[1]+g[2])/C[1], g[2]/C[1]],
                     [0, g[2]/C[2], -(g[2]+g[3]+gp)/C[2]]])
B_vec = np.array([g[0]/C[0], 0, 0])

def simula(sp, so0, Kp, Ti, Td, dt=0.05, mu=0.1, alpha=0.0, modo='PI_D',
           sat=False, aw='none', gp=0.0, gp_t=None, Kaw=1.0):
    """sp: vetor do SP (uma amostra por segundo). so0: SC inicial [%] (regime). Retorna t, VP, SC."""
    Ki, Kd, Tf = Kp/Ti, Kp*Td, mu*Td
    T = len(sp); n = int(T/dt); t = np.arange(n)*dt
    A = A_mat(); x = np.linalg.solve(-A, B_vec)*so0
    def disc(Am):
        Phi = expm(Am*dt); return Phi, np.linalg.solve(Am, Phi-np.eye(3)) @ B_vec
    Phi, Gam = disc(A)
    Phi2, Gam2 = disc(A_mat(gp)) if gp > 0 else (Phi, Gam)
    y0 = x[2]; I = so0 if modo == 'PI_D' else so0 + Kp*y0
    xf = y0; xs = sp[0]; VP = np.zeros(n); SC = np.zeros(n)
    for k in range(n):
        spk = sp[min(int(t[k]), T-1)]
        if gp_t is not None and t[k] >= gp_t: Phi, Gam = Phi2, Gam2
        if alpha > 0:                                  # filtro 2GL: ((1-a)Ti s+1)/(Ti s+1)
            xs += (1-np.exp(-dt/Ti))*(spk-xs); spf = (1-alpha)*spk + alpha*xs
        else: spf = spk
        y = x[2]; xf += (1-np.exp(-dt/Tf))*(y-xf); der = Kd*(y-xf)/Tf
        e = spf - y
        u = (Kp*e + I - der) if modo == 'PI_D' else (I - Kp*y - der)
        uc = min(max(u, 0), 100) if sat else u
        if aw == 'clamp':
            if not ((u > 100 and e > 0) or (u < 0 and e < 0)): I += Ki*e*dt
        elif aw == 'backcalc': I += (Ki*e + Kaw*(uc-u))*dt
        else: I += Ki*e*dt
        VP[k] = y; SC[k] = uc if sat else u
        x = Phi @ x + Gam*uc
    return t, VP, SC

def le_csv(nome):
    """Mesma logica do script do professor: 5 variaveis empilhadas (SO,SP,VP,VP1,VP2) na 3a coluna."""
    vals = []
    for ln in open(nome, encoding='utf-8', errors='ignore'):
        ln = ln.strip()
        if not ln: continue
        for sep in (';', '\t', ','):
            if sep in ln: break
        cols = ln.split(sep) if sep != ',' or ln.count(',') <= 2 else ln.split(',')
        if len(cols) < 3: continue
        try: vals.append(float(cols[2].replace(',', '.').replace('"', '')))
        except ValueError: pass
    m = np.array(vals); div = len(m)//5
    return [m[i*div:(i+1)*div] for i in range(5)]      # SO, SP, VP, VP1, VP2

def metricas(a, b):
    d = a - b; return np.sqrt(np.mean(d**2)), np.max(np.abs(d))

def autoteste():
    sp = np.where(np.arange(600) < 50, 20.0, 30.0)
    t, vp, sc = simula(sp, 20/0.397146, 3.523, 90.99, 11.114)
    print("Item 1.1, FT completa: VP final %.2f | pico SC %.1f (esperado 87.8) | SC final %.1f (esperado 75.5)" % (vp[-1], sc.max(), sc[-1]))
    sp = np.where(np.arange(600) < 50, 10.0, 12.0)
    t, vp, sc = simula(sp, 10/0.397146, 28.654, 43.031, 6.885)
    print("Item 4.1, FT completa: sobressinal %.0f %% (esperado 62-65)" % ((vp.max()-12)/2*100))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('csv', nargs='?')
    ap.add_argument('--kp', type=float); ap.add_argument('--ti', type=float); ap.add_argument('--td', type=float)
    ap.add_argument('--mu', type=float, default=0.1); ap.add_argument('--alpha', type=float, default=0.0)
    ap.add_argument('--modo', default='PI_D'); ap.add_argument('--sat', action='store_true'); ap.add_argument('--aw', default='none')
    ap.add_argument('--rp', type=float); ap.add_argument('--rp-t', type=float); ap.add_argument('--inicio', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'autoteste': autoteste(); sys.exit()
    SO, SP, VP, VP1, VP2 = [v[a.inicio:] for v in le_csv(a.csv)]
    gp = 1/a.rp if a.rp else 0.0
    t, vps, scs = simula(SP, SO[0], a.kp, a.ti, a.td, mu=a.mu, alpha=a.alpha, modo=a.modo, sat=a.sat, aw=a.aw, gp=gp, gp_t=a.rp_t)
    idx = (np.arange(len(SO))/0.05).astype(int); idx = np.minimum(idx, len(vps)-1)
    r1 = metricas(vps[idx], VP); r2 = metricas(scs[idx], SO)
    print("VP: RMS %.2f %%, max %.2f %% | SC: RMS %.2f %%, max %.2f %%" % (*r1, *r2))
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(SP, 'k:', label='SP'); ax[0].plot(VP, 'b', label='VP exp.'); ax[0].plot(t, vps, 'r--', label='VP sim.'); ax[0].legend(); ax[0].grid()
    ax[1].plot(SO, 'b', label='SC exp. (SO)'); ax[1].plot(t, scs, 'r--', label='SC sim.'); ax[1].legend(); ax[1].grid()
    plt.tight_layout(); plt.savefig('comparacao.png', dpi=150); print('salvo: comparacao.png')
