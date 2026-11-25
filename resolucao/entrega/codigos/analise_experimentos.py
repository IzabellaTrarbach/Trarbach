#!/usr/bin/env python3
"""Analise dos experimentos Q11, Q12, Q13a e Q13b (ScadaBR) e comparacao com a simulacao.

Uso (a partir da pasta entrega/):  python3 codigos/analise_experimentos.py
Le  dados/Q11.csv, Q12.csv, Q13a.csv, Q13b.csv  e grava figuras em figs_exp/ e o resumo em resultados_experimentos.txt
Premissas (confirmar!): sintonia usada nos 4 ensaios = Dahlin tc=40 s (Kp=3.523, Ti=90.99 s, Td=11.114 s, mu=0.1).
"""
import sys, os, io, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from load_q import load
import simexp
from simexp import simula, amostra, mediana_movel, A, B, OFF
from scipy.linalg import expm
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 150})
P = (3.523, 90.99, 11.114)
os.makedirs('figs_exp', exist_ok=True)
LOG = io.StringIO()
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n')
D = {n: load('dados/%s.csv' % n) for n in ('Q11', 'Q12', 'Q13a', 'Q13b')}

# ---------------------------------------------------------------- 0) ganho estatico e validacao da planta
SOs = np.array([10.37, 10.37, 20.0, 50.0, 99.94]); VPs = np.array([3.81, 3.92, 7.70, 19.43, 39.39])
a, b = np.polyfit(SOs, VPs, 1)
pr("== 0) Ganho estatico (regime, manual): VP = %.4f*SO %+.3f  (teorico K3=0.3971)" % (a, b))
dt = 0.1; Phi = expm(A * dt); Gam = np.linalg.solve(A, Phi - np.eye(3)) @ B
def planta_com_SO_medido(d, a0, b0):
    x = np.array([d['VP2'][a0], d['VP1'][a0], d['VP'][a0] - OFF]); out = []
    for u in d['SO'][a0:b0]:
        for _ in range(int(1 / dt)): x = Phi @ x + Gam * u
        out.append(x[2] + OFF)
    return np.array(out)
pr("== 0b) Planta (modelo teorico) excitada com o SO MEDIDO x VP medido")
for n, a0, b0 in (('Q11', 1160, 2191), ('Q12', 20, 1075), ('Q13a', 20, 2100), ('Q13b', 20, 825)):
    d = D[n]; vp = planta_com_SO_medido(d, a0, b0); ex = d['VP'][a0 + 1:b0 + 1] if b0 < len(d['VP']) else d['VP'][a0 + 1:b0]
    m = min(len(vp), len(ex)); e = vp[:m] - ex[:m]
    pr("   %-5s RMS %.2f %%  erro max %.2f %%" % (n, np.sqrt(np.mean(e ** 2)), np.abs(e).max()))

# ---------------------------------------------------------------- 1) Q11 (item 1.1)
d = D['Q11']; SO, SP, VP = d['SO'], d['SP'], d['VP']
i0 = 1606; step = 1654; fim = len(SO)
pr("== 1) Q11 / item 1.1: auto em t=%d s (SO manual = %.1f); SP %.1f -> %.1f em t=%d s" % (i0, SO[i0 - 1], SP[step - 1], SP[step], step))
pr("   salto do SO no degrau do SP: %.1f -> %.1f  (+%.1f);  Kp*dSP = %.2f*%.2f = %.1f  => Kp efetivo = %.2f" % (
   SO[step - 1], SO[step], SO[step] - SO[step - 1], P[0], SP[step] - SP[step - 1], P[0] * (SP[step] - SP[step - 1]), (SO[step] - SO[step - 1]) / (SP[step] - SP[step - 1])))
t, vp, sc = simula(SP[i0 - 1:fim], 50.0, VP[i0 - 1], *P); n_ = fim - (i0 - 1)
vps = amostra(t, vp, n_); scs = amostra(t, sc, n_); ex = VP[i0 - 1:fim]; so_med = mediana_movel(SO[i0 - 1:fim], 21)
def metr(y, sp0, sp1, t_step):
    dd = sp1 - sp0; yn = (y - sp0) / dd; m = np.arange(len(y)) >= t_step
    t63 = next(i for i in range(t_step, len(y)) if yn[i] >= .632) - t_step
    bad = np.where(np.abs(y - sp1) > .02 * dd)[0]; ts = bad[-1] - t_step if len(bad) and bad[-1] > t_step else 0
    return max(0, (y[t_step:].max() - sp1) / dd * 100), t63, ts
ts_ = step - (i0 - 1)
pr("   VP exp: sobressinal %.1f%% t63 %d s ts(2%%, VP suavizada 15 s) %d s | VP sim: sobressinal %.1f%% t63 %d s ts %d s" % (*metr(mediana_movel(ex, 15), 19.52, 30.0, ts_), *metr(vps, 19.52, 30.0, ts_)))
pr("   VP exp bruta: max %.2f %% (sobressinal %.1f%%)" % (ex.max(), (ex.max() - 30) / 10.48 * 100))
pr("   VP: RMS %.2f %%  erro max %.2f %% | SC (mediana movel 21 s) RMS %.2f %% | SC medio final exp %.1f sim %.1f" % (
   np.sqrt(np.mean((vps - ex) ** 2)), np.abs(vps - ex).max(), np.sqrt(np.mean((scs - so_med) ** 2)), SO[-300:].mean(), scs[-300:].mean()))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4)); tt = np.arange(n_) + i0 - 1 - step
ax[0].plot(tt, SP[i0 - 1:fim], 'k:', label='SP'); ax[0].plot(tt, ex, 'b', label='VP experimental'); ax[0].plot(tt, vps, 'r--', label='VP simulada (PID Dahlin 40 s)'); ax[0].set_title('Q11 (item 1.1): VP'); ax[0].legend(fontsize=7)
ax[1].plot(tt, SO[i0 - 1:fim], color='#9ecae1', lw=.7, label='SO bruto (1 s)'); ax[1].plot(tt, so_med, 'b', label='SO mediana 21 s'); ax[1].plot(tt, scs, 'r--', label='SC simulado'); ax[1].set_title('Q11: sinal de controle'); ax[1].legend(fontsize=7)
for a_ in ax: a_.set_xlabel('t - t_degrau [s]')
plt.tight_layout(); plt.savefig('figs_exp/q11.png'); plt.close()

# ---------------------------------------------------------------- 2) ruido do SO (causa da "instabilidade")
def spikes(n, a0, b0, thr=8):
    d = D[n]; SO, VP = d['SO'][a0:b0], d['VP'][a0:b0]; base = mediana_movel(SO, 15); sp = np.abs(SO - base) > thr
    dv = np.abs(np.diff(VP, prepend=VP[0])) > 0.05; near = np.convolve(dv.astype(int), np.ones(3), 'same') > 0
    return sp.mean(), dv.mean(), sp[near].mean(), sp[~near].mean(), SO.std(), base.std(), VP.std()
pr("== 2) SO em regime com a malha FECHADA: picos x variacao de 1 LSB (0.0976 %) da VP")
for n, a0, b0 in (('Q11', 1900, 2191), ('Q13a', 100, 560), ('Q12', 620, 1075)):
    r = spikes(n, a0, b0); pr("   %-5s [%d:%d] picos %.0f%% das amostras | VP mudou 1 LSB em %.0f%% | P(pico|LSB perto)=%.2f  P(pico|sem LSB)=%.2f | sd(SO)=%.1f sd(mediana)=%.1f sd(VP)=%.2f" % (n, a0, b0, 100 * r[0], 100 * r[1], r[2], r[3], r[4], r[5], r[6]))
d = D['Q11']; SO, VP = d['SO'], d['VP']; base = mediana_movel(SO, 15)
up = [i for i in range(1900, 2190) if VP[i] - VP[i - 1] > .05]; dn_ = [i for i in range(1900, 2190) if VP[i] - VP[i - 1] < -.05]
pr("   Q11: VP subiu 1 LSB (n=%d): desvio medio do SO = %+.1f pontos | VP desceu 1 LSB (n=%d): %+.1f pontos" % (len(up), np.mean([SO[i] - base[i] for i in up]), len(dn_), np.mean([SO[i] - base[i] for i in dn_])))
Kd = P[0] * P[2]; LSB = 0.0976
pr("   Previsto com filtro mu=0.1: pico = Kd*LSB/(mu*Td) = %.1f pontos; sem filtro, derivada por diferenca com Ts=0.1 s: Kd*LSB/Ts = %.1f pontos" % (Kd * LSB / (0.1 * P[2]), Kd * LSB / 0.1))
fig, ax = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True); r = slice(2000, 2060)
ax[0].step(np.arange(2000, 2060), VP[r], 'b', where='post'); ax[0].set_ylabel('VP [%]'); ax[0].set_title('Q11, malha fechada em regime: degraus de 1 LSB (0,098 %) da VP')
ax[1].plot(np.arange(2000, 2060), SO[r], 'r.-', lw=.7); ax[1].set_ylabel('SO [%]'); ax[1].set_xlabel('t [s]'); ax[1].set_title('...e os picos correspondentes no SO')
plt.tight_layout(); plt.savefig('figs_exp/q11_ruido.png'); plt.close()

# ---------------------------------------------------------------- 3) Q12 (item 1.2, sem VP-tracking)
d = D['Q12']; SO, SP, VP = d['SO'], d['SP'], d['VP']; st = 593
pr("== 3) Q12 / item 1.2: manual SO=%.1f%% VP=%.2f%%; SP(operador)=%.1f%% (em manual o SP ficou em 0); auto em t=%d s" % (SO[st - 1], VP[st - 1], SP[st], st))
t, vp, sc = simula(SP[st - 1:], 20.0, VP[st - 1], *P); m_ = len(SP) - (st - 1); vps = amostra(t, vp, m_); scs = amostra(t, sc, m_); ex = VP[st - 1:]; so_med = mediana_movel(SO[st - 1:], 15)
pr("   SO apos a transferencia (t=593..600): %s | mediana 595-615 = %.1f" % (np.round(SO[593:601], 1), np.median(SO[595:615])))
pr("   Previsto: salto = Kp*(SP-VP) = %.2f*(%.2f-%.2f) = %.1f  => SO = %.1f" % (P[0], SP[st], VP[st - 1], P[0] * (SP[st] - VP[st - 1]), 20 + P[0] * (SP[st] - VP[st - 1])))
pr("   VP: RMS %.2f %% erro max %.2f %% | VP exp max %.2f sim max %.2f" % (np.sqrt(np.mean((vps - ex) ** 2)), np.abs(vps - ex).max(), VP[st:].max(), vps.max()))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4)); tt = np.arange(m_) + st - 1
ax[0].plot(tt - st, SP[st - 1:], 'k:', label='SP'); ax[0].plot(tt - st, ex, 'b', label='VP exp.'); ax[0].plot(tt - st, vps, 'r--', label='VP sim.'); ax[0].set_title('Q12 (item 1.2): VP'); ax[0].legend(fontsize=7); ax[0].set_xlim(-20, 480)
ax[1].plot(tt - st, SO[st - 1:], color='#9ecae1', lw=.7, label='SO bruto'); ax[1].plot(tt - st, so_med, 'b', label='SO mediana'); ax[1].plot(tt - st, scs, 'r--', label='SC sim.'); ax[1].set_title('Q12: SC na transferencia manual->auto'); ax[1].legend(fontsize=7); ax[1].set_xlim(-20, 120)
for a_ in ax: a_.set_xlabel('t - t_auto [s]')
plt.tight_layout(); plt.savefig('figs_exp/q12.png'); plt.close()

# ---------------------------------------------------------------- 4) Q13a / Q13b (item 1.3)
res = {}
for n, aw, rot in (('Q13a', 'none', 'sem anti-windup'), ('Q13b', 'clamp', 'com anti-windup')):
    d = D[n]; SO, SP, VP = d['SO'], d['SP'], d['VP']
    up = [i for i in range(1, len(SP)) if SP[i] - SP[i - 1] > 20][0]; dn = [i for i in range(1, len(SP)) if SP[i - 1] - SP[i] > 20][0]
    so_m = mediana_movel(SO, 7)
    sat_exit = next(i for i in range(dn, len(SO)) if so_m[i] < 95) - dn
    vpfall = next(i for i in range(dn, len(VP)) if VP[i] < VP[dn] - .5) - dn
    within = next(i for i in range(dn, len(VP)) if abs(VP[i] - 20) < 1.0 and np.all(np.abs(VP[i:i + 30] - 20) < 1.5)) - dn
    pr("== 4) %s (%s): SP 55 em t=%d, SP 20 em t=%d (inalcancavel por %d s); VP max %.2f%%" % (n, rot, up, dn, dn - up, VP[up:dn].max()))
    pr("   EXP: SO sai da saturacao %d s apos a queda do SP | VP comeca a cair (-0.5) em %d s | VP dentro de +-1 de 20%% em %d s | SO medio no patamar %.1f%%" % (sat_exit, vpfall, within, SO[up + 60:dn - 20].mean()))
    res[n] = dict(up=up, dn=dn, sat=sat_exit, fall=vpfall, within=within)
# simulacao: Q13a (partida em regime, SO=20), Q13b (partida em transitorio: usa os estados medidos)
d = D['Q13a']; SO, SP, VP = d['SO'], d['SP'], d['VP']; st = 820
sims = {}
for lab, aw in (('sem AW', 'none'), ('com AW', 'clamp')):
    t, vp, sc = simula(SP[st - 1:], 20.0, VP[st - 1], *P, aw=aw); m_ = len(SP) - (st - 1); sims[('Q13a', lab)] = (amostra(t, vp, m_), amostra(t, sc, m_), st - 1)
d = D['Q13b']; SO, SP, VP = d['SO'], d['SP'], d['VP']; t0 = 14; st = 38; simexp.SOMAX = 96.5
for lab, aw in (('sem AW', 'none'), ('com AW', 'clamp')):
    t, vp, sc = simula(SP[t0:], 20.0, VP[t0], *P, aw=aw, x0=[d['VP2'][t0], d['VP1'][t0], VP[t0] - OFF], pre=SO[t0:st]); m_ = len(SP) - t0; sims[('Q13b', lab)] = (amostra(t, vp, m_), amostra(t, sc, m_), t0)
simexp.SOMAX = 99.940875
for (n, lab), (vps, scs, off0) in sims.items():
    dn = res[n]['dn'] - off0; ex = D[n]['VP'][off0:]; m_ = len(vps)
    se = next((i for i in range(dn, m_) if scs[i] < 95), None); vf = next((i for i in range(dn, m_) if vps[i] < vps[dn] - .5), None)
    pr("   SIM %s / %s: SC sai da saturacao %s s apos a queda | VP cai 0.5 em %s s | VP RMS %.2f %%" % (n, lab, se - dn if se is not None else 'nunca', vf - dn if vf is not None else 'nunca', np.sqrt(np.mean((vps - ex) ** 2))))
fig, ax = plt.subplots(2, 2, figsize=(10, 5.6), sharex='col')
for j, (n, lab, cor) in enumerate((('Q13a', 'sem AW', 'r'), ('Q13b', 'com AW', 'g'))):
    d = D[n]; dn = res[n]['dn']; r = np.arange(dn - 120, min(len(d['SO']), dn + 800)); vps, scs, off0 = sims[(n, lab)]
    tt = r - dn
    ax[0, j].plot(tt, d['SP'][r], 'k:', label='SP'); ax[0, j].plot(tt, d['VP'][r], cor, label='VP exp.'); ax[0, j].plot(tt, vps[r - off0], 'k--', lw=1, label='VP sim.'); ax[0, j].set_title('%s - %s' % (n, 'sem anti reset windup' if j == 0 else 'com anti reset windup')); ax[0, j].legend(fontsize=7)
    ax[1, j].plot(tt, d['SO'][r], color='#bbbbbb', lw=.6, label='SO bruto'); ax[1, j].plot(tt, mediana_movel(d['SO'], 7)[r], cor, label='SO mediana 7 s'); ax[1, j].plot(tt, scs[r - off0], 'k--', lw=1, label='SC sim.'); ax[1, j].legend(fontsize=7); ax[1, j].set_xlabel('t - t_queda do SP [s]')
ax[0, 0].set_ylabel('VP [%]'); ax[1, 0].set_ylabel('SO [%]'); plt.tight_layout(); plt.savefig('figs_exp/q13.png'); plt.close()
fig, ax = plt.subplots(2, 1, figsize=(10, 5.4), sharex=False)
for j, n in enumerate(('Q13a', 'Q13b')):
    d = D[n]; ax[j].plot(d['t'], d['SO'], color='#fcbba1', lw=.5, label='SO'); ax[j].plot(d['t'], d['SP'], 'k:', label='SP'); ax[j].plot(d['t'], d['VP'], 'b', label='VP'); ax[j].plot(d['t'], d['VP1'], color='gold', lw=.8, label='VP1'); ax[j].plot(d['t'], d['VP2'], 'm', lw=.8, label='VP2'); ax[j].set_title(n); ax[j].legend(fontsize=7, ncol=5)
ax[1].set_xlabel('t [s]'); plt.tight_layout(); plt.savefig('figs_exp/q13_completo.png'); plt.close()

# ---------------------------------------------------------------- 5) PI x PID (simulacao na FT completa)
from scipy.signal import ss2tf
n_, d_ = ss2tf(A, B.reshape(3, 1), np.eye(3)[2:3], [[0]]); Kg = 0.397146
w = np.logspace(-3, 1, 40000); s = 1j * w; G = n_[0][-1] / np.polyval(d_, s); Gr = 0.3918 * np.exp(-25.92 * s) / (78.03 * s + 1)
def L(Kp, Ti, Td, mu, Gx): return (Kp + Kp / Ti / s + (Kp * Td * s / (mu * Td * s + 1) if Td > 0 else 0)) * Gx
def mar(Lx):
    mag = abs(Lx); ph = np.unwrap(np.angle(Lx)); i = np.where(np.diff(np.sign(mag - 1)))[0]; j = np.where(np.diff(np.sign(ph + np.pi)))[0]
    return round(180 + np.degrees(ph[i[0]]), 1), round(1 / mag[j[0]], 1)
pr("== 5) PID x PI (FT completa; degrau 20->30 %%; ganho HF e pontos de SC por 1 LSB da VP)")
casos = [("PID Dahlin mu=0.1 (usado)", (3.523, 90.99, 11.114), .1), ("PID mu=0.5", (3.523, 90.99, 11.114), .5), ("PID mu=1.0", (3.523, 90.99, 11.114), 1.0), ("PI Dahlin (Kc=3.021, Ti=78.03)", (3.021, 78.03, 0.0), .1)]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3)); spv = np.where(np.arange(600) < 50, 20.0, 30.0)
for (nm, p, mu), cor in zip(casos, ('k', 'tab:blue', 'tab:green', 'r')):
    Kp, Ti, Td = p; t, vp, sc = simula(spv, 20 / Kg, 20.0, Kp, Ti, Td, mu=mu, sat=False, aw='none', off=0.0); vps = amostra(t, vp, 600); scs = amostra(t, sc, 600)
    os_, t63, ts = metr(vps, 20.0, 30.0, 50); hf = Kp * (1 + 1 / mu) if Td > 0 else Kp
    pr("   %-32s PM %.1f GM %.1f (FOPDT: PM %.1f GM %.1f) | sobress %.1f%% t63 %d s ts %d s pico SC %.1f | ganho HF %.1f => %.1f pontos de SC por LSB" % (nm, *mar(L(Kp, Ti, Td, mu, G)), *mar(L(Kp, Ti, Td, mu, Gr)), os_, t63, ts, scs.max(), hf, hf * LSB))
    ax[0].plot(np.arange(600), vps, color=cor, label=nm); ax[1].plot(np.arange(600), scs, color=cor)
ax[0].legend(fontsize=7); ax[0].set_title('VP (simulacao, FT completa)'); ax[1].set_title('SC (simulacao)'); ax[0].set_xlabel('t [s]'); ax[1].set_xlabel('t [s]'); plt.tight_layout(); plt.savefig('figs_exp/pi_x_pid.png'); plt.close()
open('resultados_experimentos.txt', 'w').write(LOG.getvalue())
