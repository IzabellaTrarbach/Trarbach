#!/usr/bin/env python3
"""Analise do codigo ST (Prova2.st) e simulacao fiel ao PLC: reproduz o ruido do SO, os picos e os ensaios Q11/Q12/Q13.
Uso (pasta entrega/):  python3 codigos/analise_plc.py   -> figuras em figs_exp/ e resumo em resultados_plc.txt"""
import sys, os, io, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from load_q import load
from simexp import mediana_movel, OFF
from plc_sim import plc
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 150})
os.makedirs('figs_exp', exist_ok=True); LOG = io.StringIO()
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n')
KD, LSB, TS = 39.16, 0.0976, 0.02; SIG = 0.15
D = {n: load('dados/%s.csv' % n) for n in ('Q11', 'Q12', 'Q13a', 'Q13b')}
# ------------------------------------------------ 1) histograma dos picos em multiplos de Kd*LSB/(10*Ts)
unit = KD * LSB / (10 * TS); dev = []
for n, a, b in (('Q11', 1900, 2191), ('Q13a', 100, 560), ('Q12', 620, 1075)):
    SO = D[n]['SO'][a:b]; base = mediana_movel(SO, 15)
    for i in range(len(SO)):
        if 10.5 < SO[i] < 99.5 and 10.5 < base[i] < 99.5 and abs(SO[i] - base[i]) > 5: dev.append((SO[i] - base[i]) / unit)
dev = np.array(dev)
pr("== 1) Picos do SO (nao saturados) em unidades de Kd*LSB/(10*Ts) = %.2f pontos: n=%d" % (unit, len(dev)))
ks = np.arange(-4, 5); cont = {int(k): int(np.sum(np.abs(dev - k) < 0.15)) for k in ks}
pr("   contagem por multiplo inteiro:", cont, "| fracao em +-0.15 de um inteiro: %.2f (acaso: 0.30)" % np.mean(np.abs(dev - np.round(dev)) < 0.15))
fig, ax = plt.subplots(figsize=(6, 2.8)); ax.hist(dev, bins=np.arange(-4.5, 4.51, .1), color='tab:red'); ax.set_xlabel('(SO - mediana) / [Kd*LSB/(10*Ts)]'); ax.set_ylabel('amostras'); ax.set_title('Picos do SO em multiplos inteiros de %.1f pontos' % unit); plt.tight_layout(); plt.savefig('figs_exp/plc_histograma.png'); plt.close()
# ------------------------------------------------ 2) Q11 : experimento x simulacao fiel ao PLC
d = D['Q11']; SO, SP, VP = d['SO'], d['SP'], d['VP']; i0 = 1606; fim = len(SO); sp = SP[i0 - 1:fim].copy(); ts_ = 1654 - (i0 - 1)
idx = (np.arange(len(sp)) / TS).astype(int)
def met(sc, vpr, tag):
    S = sc[idx]; V = vpr[idx]; base = mediana_movel(S, 15); yn = (mediana_movel(V, 9) - 19.52) / 10.48; t63 = next(i for i in range(ts_, len(yn)) if yn[i] >= .632) - ts_
    pr("   %-36s SO medio(ult.300 s) %.1f | sd(SO) %.1f | picos %.2f | t63 %d s | VP RMS x exp %.2f" % (tag, S[-300:].mean(), S[-300:].std(), np.mean(np.abs(S[ts_ + 250:] - base[ts_ + 250:]) > 8), t63, np.sqrt(np.mean((V - VP[i0 - 1:fim]) ** 2))))
    return S, V
pr("== 2) Q11 (item 1.1): experimento x simulacao fiel ao PLC")
ex = VP[i0 - 1:fim]; base = mediana_movel(SO[i0 - 1:fim], 15); yn = (mediana_movel(ex, 9) - 19.52) / 10.48
pr("   %-36s SO medio(ult.300 s) %.1f | sd(SO) %.1f | picos %.2f | t63 %d s" % ('EXPERIMENTO', SO[-300:].mean(), SO[-300:].std(), np.mean(np.abs(SO[1654 + 250:] - mediana_movel(SO, 15)[1654 + 250:]) > 8), next(i for i in range(ts_, len(yn)) if yn[i] >= .632) - ts_))
res = {}
for nm, kw in (("PLC original (Kd=39,16 sem filtro)", {}), ("PI de Dahlin (Kp=3,021; Kd=0)", dict(kp=3.021, kd=0.0)), ("PID + passa-baixa na VP (tau=1,11 s)", dict(vpf_tau=1.11, d_init_auto=True))):
    t, sc, vpa, vpr = plc(sp, 50.0, VP[i0 - 1], sigma_lsb=SIG, **kw); res[nm] = met(sc, vpr, nm)
pr("   Sensibilidade ao ruido do ADC (PLC original):")
for sg in (0.0, 0.1, 0.15, 0.2, 0.3):
    t, sc, vpa, vpr = plc(sp, 50.0, VP[i0 - 1], sigma_lsb=sg); met(sc, vpr, "sigma = %.2f LSB" % sg)
tt = np.arange(len(sp)) - ts_
fig, ax = plt.subplots(2, 2, figsize=(10, 5.2), sharex=True)
ax[0, 0].plot(tt, SP[i0 - 1:fim], 'k:'); ax[0, 0].plot(tt, ex, 'b', label='VP exp.'); ax[0, 0].plot(tt, res["PLC original (Kd=39,16 sem filtro)"][1], 'r--', label='VP sim. PLC'); ax[0, 0].legend(fontsize=7); ax[0, 0].set_title('Q11: VP')
ax[0, 1].plot(tt, ex - res["PLC original (Kd=39,16 sem filtro)"][1][:len(ex)], 'm'); ax[0, 1].set_title('Q11: VP exp. - VP sim.')
ax[1, 0].plot(tt, SO[i0 - 1:fim], color='#9ecae1', lw=.7); ax[1, 0].set_title('SO experimental (bruto)'); ax[1, 1].plot(tt, res["PLC original (Kd=39,16 sem filtro)"][0], color='#fcae91', lw=.7); ax[1, 1].set_title('SO simulado (PLC com ADC ruidoso)')
for a_ in ax[1]: a_.set_xlabel('t - t_degrau [s]'); a_.set_ylim(0, 105)
plt.tight_layout(); plt.savefig('figs_exp/plc_q11.png'); plt.close()
# ------------------------------------------------ 3) correcoes: SO simulado
fig, ax = plt.subplots(3, 1, figsize=(9, 5.6), sharex=True)
for a_, (nm, (S, V)), c in zip(ax, res.items(), ('tab:red', 'tab:green', 'tab:blue')):
    a_.plot(tt, S, color=c, lw=.7); a_.set_title('SO simulado: ' + nm); a_.set_ylim(0, 105)
ax[2].set_xlabel('t - t_degrau [s]'); plt.tight_layout(); plt.savefig('figs_exp/plc_correcoes.png'); plt.close()
# ------------------------------------------------ 4) Q12 e Q13 com a simulacao fiel
pr("== 3) Q12 (item 1.2): auto em t=593 (SO manual 20; SP=16; VP=7,71)")
d = D['Q12']; st = 593; t, sc, vpa, vpr = plc(d['SP'][st:].copy(), 20.0, d['VP'][st], sigma_lsb=SIG); ix = (np.arange(len(d['SP']) - st) / TS).astype(int)
pr("   SO (mediana 595-615): exp %.1f  sim %.1f | VP RMS x exp %.2f | SO nos 3 primeiros scans (derivada zerada em manual): %s" % (np.median(d['SO'][595:615]), np.median(sc[ix][2:22]), np.sqrt(np.mean((vpr[ix] - d['VP'][st:]) ** 2)), np.round(sc[:4], 1)))
pr("== 4) Q13a (sem STOP_INTGR) e Q13b (com STOP_INTGR)")
fig, ax = plt.subplots(2, 2, figsize=(10, 5.6), sharex='col')
for j, (n, st, stop) in enumerate((('Q13a', 820, False), ('Q13b', 38, True))):
    d = D[n]; SO, SP, VP = d['SO'], d['SP'], d['VP']; up = [i for i in range(1, len(SP)) if SP[i] - SP[i - 1] > 20][0]; dn = [i for i in range(1, len(SP)) if SP[i - 1] - SP[i] > 20][0]
    x0 = None if n == 'Q13a' else [d['VP2'][st], d['VP1'][st], VP[st] - OFF]
    t, sc, vpa, vpr = plc(SP[st:].copy(), 20.0, VP[st], sigma_lsb=SIG, stop_aw=stop, x0=x0); ix = (np.arange(len(SP) - st) / TS).astype(int); S = sc[ix]; V = vpr[ix]; D_ = dn - st; a_ = up - st + 60; b_ = D_ - 20
    se_s = next((i for i in range(D_, len(S)) if mediana_movel(S, 7)[i] < 95), None); se_e = next(i for i in range(dn, len(SO)) if mediana_movel(SO, 7)[i] < 95) - dn
    w_s = next((i for i in range(D_, len(V)) if abs(V[i] - 20) < 1 and np.all(np.abs(V[i:i + 30] - 20) < 1.5)), None); w_e = next((i for i in range(dn, len(VP)) if abs(VP[i] - 20) < 1 and np.all(np.abs(VP[i:i + 30] - 20) < 1.5)), None)
    pr("   %s: SO medio no patamar exp %.1f sim %.1f | VPmax exp %.2f sim %.2f | SO sai da saturacao exp %d s sim %s s | VP em 20+-1: exp %s s sim %s s | VP RMS %.2f" % (n, SO[up + 60:dn - 20].mean(), S[a_:b_].mean(), VP[up:dn].max(), V[:D_].max(), se_e, None if se_s is None else se_s - D_, None if w_e is None else w_e - dn, None if w_s is None else w_s - D_, np.sqrt(np.mean((V - VP[st:]) ** 2))))
    r = np.arange(dn - 120, min(len(SO), dn + 800)); tt2 = r - dn
    ax[0, j].plot(tt2, SP[r], 'k:'); ax[0, j].plot(tt2, VP[r], 'b', label='VP exp.'); ax[0, j].plot(tt2, V[r - st], 'r--', label='VP sim. PLC'); ax[0, j].set_title('%s: %s' % (n, 'sem anti-windup' if j == 0 else 'com anti-windup (STOP_INTGR)')); ax[0, j].legend(fontsize=7)
    ax[1, j].plot(tt2, SO[r], color='#9ecae1', lw=.6, label='SO exp.'); ax[1, j].plot(tt2, S[r - st], color='#fb6a4a', lw=.6, alpha=.8, label='SO sim.'); ax[1, j].legend(fontsize=7); ax[1, j].set_xlabel('t - t_queda do SP [s]')
plt.tight_layout(); plt.savefig('figs_exp/plc_q13.png'); plt.close()
open('resultados_plc.txt', 'w').write(LOG.getvalue())
