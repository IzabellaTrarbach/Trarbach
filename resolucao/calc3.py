import numpy as np, math
from scipy.linalg import expm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.3,'figure.dpi':150})
F='figs/'
R=np.array([22000,46800,32600,66800.]);C=np.array([947e-6,947e-6,955e-6]);g=1/R
def Amat(gp=0.0):
    gg=g.copy()
    return np.array([[-(gg[0]+gg[1])/C[0], gg[1]/C[0],0],[gg[1]/C[1],-(gg[1]+gg[2])/C[1],gg[2]/C[1]],[0,gg[2]/C[2],-(gg[2]+gg[3]+gp)/C[2]]])
Bv=np.array([g[0]/C[0],0,0]); dt=0.05
def disc(A):
    Phi=expm(A*dt);Gam=np.linalg.solve(A,Phi-np.eye(3))@Bv;return Phi,Gam
D0=disc(Amat());
K3=0.397146
def run(Kp,Ti,Td,spf,tend=600,mode='PI_D',sat=False,aw='none',mu=0.1,gp_t=None,gp=0.0,auto_t=0,sc_man=None,track=True,x0=None,Kaw=1.0):
    Ki=Kp/Ti;Kd=Kp*Td;Tf=mu*Td
    N=int(tend/dt);t=np.arange(N)*dt
    sp0=spf(0)
    A0=Amat();x=np.linalg.solve(-A0,Bv)*(sc_man if sc_man is not None else sp0/K3)
    Phi,Gam=D0;Phi2,Gam2=disc(Amat(gp)) if gp else D0
    I=(sp0/K3 if sc_man is None else sc_man);I=I if mode=='PI_D' else I+Kp*x[2];xf=x[2];ys=np.zeros(N);us=np.zeros(N);sps=np.zeros(N)
    u_man=sc_man if sc_man is not None else sp0/K3
    for k in range(N):
        y=x[2];sp=spf(t[k])
        auto=t[k]>=auto_t
        if gp_t is not None and t[k]>=gp_t: Phi,Gam=Phi2,Gam2
        xf+=dt/Tf*(y-xf)
        if not auto:
            u=u_man
            if track: sp=y;sps[k]=sp
            else: sps[k]=sp
            I=u if mode=='PI_D' else u+Kp*y
            # for PI_D: u=Kp e + I - D(y); keep I so that u=u_man when e=0 and D=0
        else:
            e=sp-y;dterm=Kd*(y-xf)/Tf
            if mode=='PI_D': u=Kp*e+I-dterm
            else: u=I-Kp*y-dterm   # I_PD: I integrates e; P and D on PV
            sps[k]=sp
        us[k]=u;uc=min(max(u,0),100) if sat else u
        if auto:
            e=sp-y
            if aw=='none': I+=Ki*e*dt
            elif aw=='clamp':
                if not ((u>100 and e>0) or (u<0 and e<0)): I+=Ki*e*dt
            elif aw=='backcalc': I+=(Ki*e+Kaw*(uc-u))*dt
        else: pass
        x=Phi@x+Gam*uc;ys[k]=y
    return t,ys,us,sps
dah=(3.523,90.99,11.114)
print("== 1.3 anti-windup")
spf=lambda t: 25 if t<50 else (45 if t<350 else 30)
for aw in ('none','clamp'):
    t,y,u,_=run(*dah,spf,tend=1000,sat=True,aw=aw)
    m=t>=350;ok=(np.abs(y-30)<0.5)&m;i=np.argmax(ok) if ok.any() else None
    print(aw,"VP max",y.max().round(2),"t to |VP-30|<0.5 after SP change:",None if i is None else (t[i]-350).round(1),"u_int at 350:",u[int(350/dt)].round(1),"u_int at 600",u[int(600/dt)].round(1),"VP@400",y[int(400/dt)].round(1),"VP@500",y[int(500/dt)].round(1))
    np.save('aw_%s.npy'%aw,np.vstack([t,y,u]))
fig,ax=plt.subplots(1,2,figsize=(9,3))
for aw,c,l in (('none','r','sem anti-windup'),('clamp','g','com anti-windup')):
    t,y,u=np.load('aw_%s.npy'%aw);ax[0].plot(t,y,c,label=l);ax[1].plot(t,np.clip(u,0,100),c)
ax[0].plot(t,[spf(x) for x in t],'k:',label='SP');ax[0].legend(fontsize=7);ax[0].set_title('VP [%]');ax[1].set_title('SC aplicado [%]');ax[0].set_xlabel('t [s]');ax[1].set_xlabel('t [s]');plt.tight_layout();plt.savefig(F+'g13.png');plt.close()
print("== 1.2 bumpless")
for tr,spf2 in ((False,lambda t:30),(True,lambda t: 20.02 if t<150 else 30)):
    t,y,u,sps=run(*dah,spf2,tend=450,auto_t=50,sc_man=50.4,track=False)
    print("track",tr,"SC@49:",u[int(49/dt)].round(1),"SC@50.1:",u[int(50.1/dt)].round(1),"VP@49",y[int(49/dt)].round(2),"VP@100",y[int(100/dt)].round(2),"SC@100",u[int(100/dt)].round(1),"SC max",u.max().round(1))
    np.save('bl_%d.npy'%tr,np.vstack([t,y,u,sps]))
fig,ax=plt.subplots(1,2,figsize=(9,3))
for tr,c,l in ((0,'r','sem VP-tracking (SP=30 fixo)'),(1,'g','com VP-tracking')):
    t,y,u,sps=np.load('bl_%d.npy'%tr);ax[0].plot(t,y,c,label=l);ax[1].plot(t,u,c)
ax[0].plot(t,[30]*len(t),'k:',label='SP = 30 (sem tracking)');ax[0].plot(t,[20.02 if x<150 else 30 for x in t],'b:',label='SP (com tracking)');ax[0].legend(fontsize=7);ax[0].set_title('VP [%]');ax[1].set_title('SC [%]');ax[0].set_xlabel('t [s]');ax[1].set_xlabel('t [s]');plt.tight_layout();plt.savefig(F+'g12.png');plt.close()
print("== 2.2 Rp")
Rp=2*R[3];gp=1/Rp;print("Rp",Rp,"gp",gp*1e6)
A1=Amat(gp);Kn=np.linalg.solve(-A1,Bv)[2];print("K3 with Rp",Kn,"ratio",Kn/K3)
print("Rt eq:", "steady V3 ratio check via resistors")
sp=25;sc0=sp/K3;print("SC0",sc0,"SC needed with Rp",sp/Kn,"open loop VP after Rp",Kn*sc0)
reg=(5.057,90.99,11.114);serv=(2.234,90.99,11.114);dh=dah
fig,ax=plt.subplots(1,2,figsize=(9,3))
for nm,p,c in (("servo (τc=τ)",serv,'g'),("regulatório τc=20 / 2GL",reg,'r'),("Dahlin τc=40",dh,'b')):
    t,y,u,_=run(*p,lambda t:sp,tend=900,gp_t=50,gp=gp)
    dip=sp-y.min();ok=(np.abs(y-sp)<1.0)&(t>t[np.argmin(y)]);i=np.argmax(ok) if ok.any() else None
    print(nm,"dip %.2f"%dip,"t_min %.0f"%(t[np.argmin(y)]-50),"back within 1.0:",None if i is None else round(t[i]-50),"SC final %.1f"%u[-1],"SC max %.1f"%u.max(),"VP end %.2f"%y[-1])
    ax[0].plot(t,y,c,label=nm);ax[1].plot(t,u,c)
t,y,u,_=run(*reg,lambda t:sp,tend=600)
ax[0].axhline(Kn*sc0,color='gray',ls=':',label='malha aberta após Rp');ax[0].legend(fontsize=7);ax[0].set_title('VP [%] (Rp ligado em t=50 s)');ax[1].set_title('SC [%]');ax[1].axhline(100,color='gray',ls=':');plt.tight_layout();plt.savefig(F+'g22.png');plt.close()
print("== 4.2 I_PD")
zn=(28.654,43.031,6.885)
spz=lambda t: 10 if t<50 else 12
fig,ax=plt.subplots(1,2,figsize=(9,3))
for md,c,l in (('PI_D','k','PI_D'),('I_PD','r','I_PD')):
    t,y,u,_=run(*zn,spz,tend=600,mode=md);print("ZN",md,"VP max %.2f"%y.max(),"os %.1f"%((y.max()-12)/2*100),"SC max %.1f SC min %.1f"%(u.max(),u.min()),"t63",t[np.argmax((y>10+1.264)&(t>50))]-50)
    ax[0].plot(t,y,c,label=md);ax[1].plot(t,u,c)
ax[0].plot(t,[spz(x) for x in t],'b:',label='SP');ax[0].legend();ax[0].set_title('VP (Ziegler–Nichols)');ax[1].set_title('SC');ax[1].axhline(0,color='gray',ls=':');plt.tight_layout();plt.savefig(F+'g42.png');plt.close()
spd=lambda t: 20 if t<50 else 30
for md in ('PI_D','I_PD'):
    t,y,u,_=run(*dah,spd,tend=800,mode=md);print("Dahlin",md,"os %.1f"%max(0,(y.max()-30)/10*100),"t63 %.0f"%(t[np.argmax((y>26.32)&(t>50))]-50),"SC max %.1f"%u.max(),"SC@50.1 %.1f"%u[int(50.1/dt)])
