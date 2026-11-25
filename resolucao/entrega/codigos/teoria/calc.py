import numpy as np, math
from scipy.signal import lsim, ss2tf, TransferFunction as TF, lti
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.3,'figure.dpi':150})
F='figs/'
R=np.array([22000,46800,32600,66800.]);C=np.array([947e-6,947e-6,955e-6]);g=1/R
A=np.array([[-(g[0]+g[1])/C[0], g[1]/C[0],0],[g[1]/C[1],-(g[1]+g[2])/C[1],g[2]/C[1]],[0,g[2]/C[2],-(g[2]+g[3])/C[2]]]);Bv=np.array([g[0]/C[0],0,0])
def fulltf(j):
    n,d=ss2tf(A,Bv.reshape(3,1),np.eye(3)[j:j+1],[[0]]);n=n[0]
    n=np.trim_zeros(np.where(abs(n)<1e-12,0,n),'f'); 
    if j==2: n=np.array([n[-1]])
    return n,d
FT=[fulltf(j) for j in range(3)]
for j in range(3): print("full",j+1,FT[j][0]/FT[j][1][-1],FT[j][1]/FT[j][1][-1])
red={0:(0.8671,23.31,0.0),1:(0.5861,68.88,9.38),2:(0.3918,78.03,25.92)}
# residues c_i
tau=np.array([68.68216,15.99884,10.38980]);
zs={0:[52.8494,11.9290],1:[20.9224],2:[]};K=[0.869203,0.590963,0.397146]
print("--- residues")
for j in range(3):
    ci=[]
    for i in range(3):
        num=np.prod([1-z/tau[i] for z in zs[j]]) if zs[j] else 1.0
        den=np.prod([1-tau[k]/tau[i] for k in range(3) if k!=i])
        ci.append(num/den)
    print("Vc%d"%(j+1),np.round(ci,4),sum(ci))
# step compare full vs red
t=np.linspace(0,400,4001)
fig,ax=plt.subplots(1,3,figsize=(9,2.8))
def padeTF(K,tau_,t0,n=3):
    from math import factorial as f
    m=n; c=[f(2*m-k)*f(m)/(f(2*m)*f(k)*f(m-k)) for k in range(m+1)]
    pn=[ (-1)**k*c[k]*t0**k for k in range(m+1)][::-1]; pd=[c[k]*t0**k for k in range(m+1)][::-1]
    return np.convolve([K],pn),np.convolve([tau_,1],pd)
for j in range(3):
    n,d=FT[j];_,y=lti(n,d).step(T=t);_,yr=lti(*padeTF(*red[j])).step(T=t) if red[j][2]>0 else lti([red[j][0]],[red[j][1],1]).step(T=t)
    ax[j].plot(t,y/y[-1],'k',label='completa');ax[j].plot(t,yr/yr[-1],'r--',label='reduzida');ax[j].set_title('Vc%d/Vin (normalizada)'%(j+1));ax[j].set_xlabel('t [s]')
ax[0].legend();plt.tight_layout();plt.savefig(F+'steps.png');plt.close()
# Bode
w=np.logspace(-3,.5,3000);s=1j*w
fig,ax=plt.subplots(2,3,figsize=(9,4.5))
Gf=[np.polyval(FT[j][0],s)/np.polyval(FT[j][1],s) for j in range(3)]
Gr=[red[j][0]*np.exp(-red[j][2]*s)/(red[j][1]*s+1) for j in range(3)]
for j in range(3):
    ax[0,j].semilogx(w,20*np.log10(abs(Gf[j])),'k',w,20*np.log10(abs(Gr[j])),'r--');ax[0,j].set_title('Vc%d/Vin'%(j+1))
    ax[1,j].semilogx(w,np.degrees(np.unwrap(np.angle(Gf[j]))),'k',w,np.degrees(-np.arctan(w*red[j][1])-w*red[j][2]),'r--');ax[1,j].axhline(-180,color='b',lw=.6,ls=':');ax[1,j].set_xlabel('ω [rad/s]')
ax[0,0].set_ylabel('módulo [dB]');ax[1,0].set_ylabel('fase [°]');plt.tight_layout();plt.savefig(F+'bode.png');plt.close()
# worked numbers for Bode & margins
print("--- Vc3 w=0.1")
for T in tau: print(T,0.1*T,math.sqrt(1+(0.1*T)**2),math.degrees(math.atan(0.1*T)))
print("red:",0.1*78.03,math.sqrt(1+(.1*78.03)**2),math.degrees(math.atan(.1*78.03)),.1*25.92,math.degrees(.1*25.92))
print("--- margins")
# Routh
a3,a2,a1=11416.68,1978.65,95.0708
print("Kcu",(a2*a1/a3-1)/0.39715,"wu",math.sqrt(a1/a3),"Tu",2*math.pi/math.sqrt(a1/a3), a2*a1/a3)
# bisection red
def f(wv): return -math.atan(wv*78.03)-wv*25.92+math.pi
lo,hi=0.01,0.2
for k in range(8):
    mid=(lo+hi)/2;print(k,lo,hi,mid,f(mid));
    if f(mid)>0: lo=mid
    else: hi=mid
w180=(lo+hi)/2
for _ in range(60):
    mid=(lo+hi)/2
    if f(mid)>0: lo=mid
    else: hi=mid
w180=(lo+hi)/2;print("w180red",w180,math.sqrt(1+(w180*78.03)**2)/0.3918)
from scipy.optimize import brentq
for nm,Gfun in [("Vc3 full",lambda wv: np.polyval(FT[2][0],1j*wv)/np.polyval(FT[2][1],1j*wv)),("Vc3 red",lambda wv:0.3918*np.exp(-25.92j*wv)/(78.03j*wv+1)),("Vc2 full",lambda wv: np.polyval(FT[1][0],1j*wv)/np.polyval(FT[1][1],1j*wv)),("Vc2 red",lambda wv:0.5861*np.exp(-9.38j*wv)/(68.88j*wv+1)),("Vc1 full",lambda wv: np.polyval(FT[0][0],1j*wv)/np.polyval(FT[0][1],1j*wv)),("Vc1 red",lambda wv:0.8671/(23.31j*wv+1))]:
    for Kc in (3,5,10):
        wg=brentq(lambda x:Kc*abs(Gfun(x))-1,1e-4,5)
        ph=np.degrees(np.unwrap(np.angle(Gfun(np.logspace(-4,np.log10(wg),4000))))[-1])
        print(nm,"Kc",Kc,"wg=%.5f |G|=%.4f phase=%.2f PM=%.1f"%(wg,abs(Gfun(wg)),ph,180+ph))
