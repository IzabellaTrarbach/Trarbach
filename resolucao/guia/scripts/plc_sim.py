"""Simulador FIEL ao programa ST do OpenPLC (Prova2.st): scan de 20 ms, ADC de 10 bits com ruido,
derivada de 4 pontos sem filtro (bloco DERIVATIVE), integral com STOP_INTGR, bias SO_AUX, LIMIT(10,u,100),
deadband de 0,09 e PID 2GL por FF2GL. Planta: circuito RC completo (3a ordem) + offset de saida."""
import numpy as np
from scipy.linalg import expm
from simexp import A, B, OFF
LSB = 100.0 / 1024.0                      # 0,0977 % (o programa usa 0,001525*64 = 0,0976)
def plc(SP, so0, vp0, kp=3.523, ki=0.03872, kd=39.16, alfa=1.0, Ts=0.02, sigma_lsb=0.25, stop_aw=True,
        umin=10.0, umax=100.0, dead=0.09, seed=1, x0=None, Rp_gp=0.0, gp_t=None, vpf_tau=0.0, d_init_auto=False):
    """SP: vetor 1 amostra/s (valor desejado enviado pelo operador). so0: SO manual (bias). vp0: VP em regime.
    stop_aw=False emula o programa SEM anti-windup (STOP_INTGR forcado a 0). Retorna (t, SC, VP_adc, VP_real).
    d_init_auto=False reproduz o programa original (derivada zerada em manual => 'chute' de 3 scans ao fechar a malha).
    vpf_tau>0 aplica um passa-baixa na VP antes da derivada (correcao proposta)."""
    rng = np.random.default_rng(seed)
    n = int(len(SP) / Ts); Phi = expm(A * Ts); Gam = np.linalg.solve(A, Phi - np.eye(3)) @ B
    Phi2 = Gam2 = None
    if Rp_gp > 0:
        A2 = A.copy(); A2[2, 2] -= Rp_gp / 955e-6; Phi2 = expm(A2 * Ts); Gam2 = np.linalg.solve(A2, Phi2 - np.eye(3)) @ B
    x = np.linalg.solve(-A, B) * so0; x = x * (vp0 - OFF) / x[2]
    if x0 is not None: x = np.array(x0, float)
    I = 0.0                                # INTEGRAL_SEM_CI (zera em manual); o bias entra por SO_PERC
    X1 = X2 = X3 = -kd * vp0               # estados do bloco DERIVATIVE
    stop = False; ff = 0.0; sp_ant = SP[0]; vpf = vp0
    if not d_init_auto: X1 = X2 = X3 = 0.0     # no programa original a derivada e zerada em manual (SEL(MF,0,...)): chute ao fechar a malha
    sc = np.zeros(n); vpa = np.zeros(n); vpr = np.zeros(n)
    for k in range(n):
        spk = SP[min(int(k * Ts), len(SP) - 1)]
        y = x[2] + OFF
        vp = round((y + sigma_lsb * LSB * rng.standard_normal()) / LSB) * LSB       # ADC de 10 bits com ruido
        e = spk - vp
        if abs(e) < dead: e = 0.0
        xin_i = 0.0 if stop else ki * e
        I += xin_i * Ts                                                           # integral (CYCLE*0,04 = Ts)
        vpf += (Ts / (vpf_tau + Ts)) * (vp - vpf) if vpf_tau > 0 else (vp - vpf)   # filtro opcional da VP (so p/ a derivada)
        xin_d = -kd * vpf
        d = (3.0 * (xin_d - X3) + X1 - X2) / (10.0 * Ts)                          # DERIVATIVE de 4 pontos (x25 do programa)
        X3, X2, X1 = X2, X1, xin_d
        if spk != sp_ant: ff += kp * (1.0 - alfa) * (spk - sp_ant); sp_ant = spk   # FF2GL
        u = kp * e + I + d + so0 - ff
        stop = (u >= umax or u <= umin) if stop_aw else False
        uc = min(max(u, umin), umax)
        if gp_t is not None and k * Ts >= gp_t: Phi, Gam = Phi2, Gam2
        x = Phi @ x + Gam * uc
        sc[k] = uc; vpa[k] = vp; vpr[k] = y
    return np.arange(n) * Ts, sc, vpa, vpr
