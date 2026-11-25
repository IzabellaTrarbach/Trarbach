import numpy as np, math
exec(open('calc.py').read().split("# step compare")[0])
from math import factorial as fa
def pade(t0,n=3):
    c=[fa(2*n-k)*fa(n)/(fa(2*n)*fa(k)*fa(n-k)) for k in range(n+1)]
    return [(-1)**k*c[k]*t0**k for k in range(n+1)][::-1],[c[k]*t0**k for k in range(n+1)][::-1]
K3,TAU,T0=0.3918,78.03,25.92
pn,pd=pade(T0);print("pade num",pn,"den",pd)
GFo=(np.convolve([K3],pn),np.convolve([TAU,1],pd)); GFu=FT[2]
def padd(a,b):
    n=max(len(a),len(b));return np.pad(a,(n-len(a),0))+np.pad(b,(n-len(b),0))
def ctrl(Kp,Ti,Td,mu=.1):
    Ki=Kp/Ti;Kd=Kp*Td
    Cn=padd(np.convolve([Kp,Ki],[mu*Td,1]),np.convolve([Kd,0],[1,0]));Cd=np.convolve([1,0],[mu*Td,1]);return Cn,Cd
def cl(Kp,Ti,Td,G,sp0,sp1,K,tend=600,alpha=0,tstep=50,mu=.1,dt=.05):
    Cn,Cd=ctrl(Kp,Ti,Td,mu);Gn,Gd=G
    den=padd(np.convolve(Cd,Gd),np.convolve(Cn,Gn))
    PIn=np.convolve([Kp,Kp/Ti],[mu*Td,1])
    Yn=np.convolve(PIn,Gn);Un=np.convolve(PIn,Gd);den2=den
    if alpha>0:
        Yn=np.convolve(Yn,[(1-alpha)*Ti,1]);Un=np.convolve(Un,[(1-alpha)*Ti,1]);den2=np.convolve(den,[Ti,1])
    t=np.arange(0,tend,dt);r=np.where(t>=tstep,sp1-sp0,0.)
    _,y,_=lsim(TF(Yn,den2),r,t);_,u,_=lsim(TF(Un,den2),r,t);return t,y+sp0,u+sp0/K,den
def st(t,y,u,sp0,sp1,ts_=50):
    d=sp1-sp0;yn=(y-sp0)/d;m=t>=ts_;bad=np.where(abs(y-sp1)>.02*d)[0]
    return dict(os=max(0,(y.max()-sp1)/d*100),t63=t[np.argmax((yn>=.632)&m)]-ts_,ts=t[bad[-1]]-ts_,upk=u.max(),uf=u[-1])
def ideal(tc,K=K3,tau=TAU,t0=T0): return (2*tau+t0)/(2*K*(tc+t0)),tau+t0/2,tau*t0/(2*tau+t0)
print("==1.1")
print("series:",TAU/(K3*(T0+40)),TAU,T0/2)
Kp,Ti,Td=ideal(40);print(Kp,Ti,Td,Kp/Ti,Kp*Td,.1*Td)
for tc in (20,30,40,60):
    row=[]
    for G in (GFo,GFu):
        t,y,u,den=cl(*ideal(tc),G,20,30,K3 if G is GFo else .397146);s=st(t,y,u,20,30);row.append("%.1f/%.1f %.1f/%.1f %.0f/%.0f %.1f/%.1f"%(s['os'],0,s['t63'],0,s['ts'],0,s['upk'],0))
    print(tc,ideal(tc),row)
t,yo,uo,den=cl(Kp,Ti,Td,GFo,20,30,K3);t,yu,uu,den2=cl(Kp,Ti,Td,GFu,20,30,.397146)
print("den FOPDT cl",den/den[0],"roots",np.roots(den));print("den full cl roots",np.roots(den2))
fig,ax=plt.subplots(1,2,figsize=(9,3))
for tc,c in zip((20,30,40,60),'rgbk'):
    t,y,u,_=cl(*ideal(tc),GFo,20,30,K3);ax[0].plot(t,y,c,label='τc=%d s'%tc);ax[1].plot(t,u,c)
ax[0].set_title('VP (FOPDT)');ax[1].set_title('SC (FOPDT)');ax[1].axhline(100,color='gray',ls=':');ax[0].legend();ax[0].set_xlabel('t [s]');ax[1].set_xlabel('t [s]');plt.tight_layout();plt.savefig(F+'i11_tc.png');plt.close()
fig,ax=plt.subplots(1,2,figsize=(9,3))
ax[0].plot(t,np.where(t>=50,30,20),'k:',label='SP');ax[0].plot(t,yo,'r--',label='FOPDT');ax[0].plot(t,yu,'k',label='FT completa');ax[1].plot(t,uo,'r--');ax[1].plot(t,uu,'k');ax[1].axhline(100,color='gray',ls=':')
ax[0].set_title('VP');ax[1].set_title('SC');ax[0].legend();plt.tight_layout();plt.savefig(F+'i11.png');plt.close()
for nm,G,K in (("FOPDT",GFo,K3),("full",GFu,.397146)):
    s=st(t,*[yo,uo] if nm=="FOPDT" else [yu,uu],20,30);print(nm,s)
print("==2.1")
reg=ideal(20);serv=ideal(TAU);print("reg",reg,reg[0]/reg[1],reg[0]*reg[2],"serv",serv,serv[0]/serv[1],serv[0]*serv[2])
print("K'c reg",TAU/(K3*(T0+20)),"serv",TAU/(K3*(T0+TAU)))
t,yS,uS,_=cl(*serv,GFo,20,30,K3);best=None;sweep=[]
for a in np.arange(0,1.0,.01):
    t,y,u,_=cl(*reg,GFo,20,30,K3,alpha=a if a>0 else 0);m=(t>50)&(t<550);e=100*np.sqrt(np.mean((y[m]-yS[m])**2))/10;sweep.append((a,e))
    if abs(a-.4)<1e-9 or abs(a-.64)<1e-9 or abs(a-.7)<1e-9 or abs(a-.99)<1e-9 or a==0: print("a=%.2f eps=%.2f"%(a,e),st(t,y,u,20,30))
sw=np.array(sweep);print("best",sw[np.argmin(sw[:,1])])
fig,ax=plt.subplots(1,2,figsize=(9,3))
ax[0].plot(t,np.where(t>=50,30,20),'k:');ax[0].plot(t,yS,'g',label='servo τc=τ');t_,y_,u_,_=cl(*reg,GFo,20,30,K3);ax[0].plot(t,y_,'r:',label='regulatório τc=20');ax[1].plot(t,uS,'g');ax[1].plot(t,u_,'r:')
for a,c in ((.4,'orange'),(.7,'purple'),(.99,'brown'),(.64,'k')):
    t_,y_,u_,_=cl(*reg,GFo,20,30,K3,alpha=a);ax[0].plot(t,y_,c,ls='--',label='2GL α=%.2f'%a);ax[1].plot(t,u_,c,ls='--')
ax[1].axhline(100,color='gray',ls=':');ax[0].legend(fontsize=6);ax[0].set_title('VP');ax[1].set_title('SC');plt.tight_layout();plt.savefig(F+'i21.png');plt.close()
plt.figure(figsize=(4.5,2.8));plt.plot(sw[:,0],sw[:,1]);plt.axvline(.64,color='r',ls=':');plt.xlabel('α');plt.ylabel('RMS vs servo [%]');plt.tight_layout();plt.savefig(F+'i21_alpha.png');plt.close()
t,y2,u2,_=cl(*reg,GFo,20,30,K3,alpha=.64);t,y3,u3,_=cl(*reg,GFu,20,30,.397146,alpha=.64)
print("2GL .64 FOPDT",st(t,y2,u2,20,30),"full",st(t,y3,u3,20,30))
print("==3.1")
r=T0/TAU;print("r",r,"ln r",math.log(r))
for nm,(c1,e1,c2,e2,c3,e3) in {"IAE servo":(1.086,-.869,None,None,.348,.914),"dist":(1.435,-.921,None,None,.482,1.137)}.items():
    print(nm,"r^e1",r**e1,"ln",e1*math.log(r),"r^e3",r**e3,e3*math.log(r))
r1=r**-.869;Kp3=1.086/K3*r1;print("1.086/K",1.086/K3,"Kp",Kp3);Ti3=TAU/(.74-.13*r);print("den",.74-.13*r,"Ti",Ti3);Td3=.348*TAU*r**.914;print("Td",Td3,.348*TAU,r**.914,"Ki",Kp3/Ti3,"Kd",Kp3*Td3)
Kd_=1.435/K3*r**-.921;print("dist: 1.435/K",1.435/K3,"r^-.921",r**-.921,"Kp",Kd_,"r^.749",r**.749,"Ti",TAU/.878*r**.749,"r^1.137",r**1.137,"Td",.482*TAU*r**1.137,"Ki",Kd_/(TAU/.878*r**.749),"Kd",Kd_*.482*TAU*r**1.137)
p31=(Kp3,Ti3,Td3);p32=(Kd_,TAU/.878*r**.749,.482*TAU*r**1.137)
fig,ax=plt.subplots(1,2,figsize=(9,3));t=np.arange(0,600,.05);ax[0].plot(t,np.where(t>=50,25,20),'k:',label='SP')
for nm,p,G,K,c,ls in (("IAE servo FOPDT",p31,GFo,K3,'r','--'),("IAE servo completa",p31,GFu,.397146,'k','-')):
    t,y,u,_=cl(*p,G,20,25,K);ax[0].plot(t,y,c,ls=ls,label=nm);ax[1].plot(t,u,c,ls=ls);print(nm,st(t,y,u,20,25))
for nm,p,G,K in (("dist FOPDT",p32,GFo,K3),("dist full",p32,GFu,.397146)):
    t,y,u,_=cl(*p,G,20,25,K);print(nm,st(t,y,u,20,25))
t,y,u,_=cl(*ideal(40),GFo,20,25,K3);print("dahlin",st(t,y,u,20,25));ax[0].plot(t,y,'g:',label='Dahlin τc=40');ax[1].plot(t,u,'g:')
ax[1].axhline(100,color='gray',ls=':');ax[0].legend(fontsize=6);ax[0].set_title('VP');ax[1].set_title('SC');plt.tight_layout();plt.savefig(F+'i31.png');plt.close()
w=np.logspace(-3,.5,40000);s=1j*w
Gred=K3*np.exp(-T0*s)/(TAU*s+1);Gfull=np.polyval(FT[2][0],s)/np.polyval(FT[2][1],s)
def L(p,G,mu=.1):
    Kp,Ti,Td=p;return (Kp+Kp/Ti/s+Kp*Td*s/(mu*Td*s+1))*G
def mar(Lx):
    mag=abs(Lx);ph=np.unwrap(np.angle(Lx));i=np.where(np.diff(np.sign(mag-1)))[0];j=np.where(np.diff(np.sign(ph+np.pi)))[0]
    return [(round(w[k],4),round(180+np.degrees(ph[k]),1)) for k in i],[(round(w[k],4),round(1/mag[k],2)) for k in j]
print("margins 1.1",mar(L(ideal(40),Gred)),mar(L(ideal(40),Gfull)));print("reg",mar(L(reg,Gred)),mar(L(reg,Gfull)));print("3.1",mar(L(p31,Gred)),mar(L(p31,Gfull)))
print("==4.1")
Kcu,Tu=38.97,68.85;Kc_=Kcu/1.7;Ti_=Tu/2;Td_=Tu/8;print(Kc_,Ti_,Td_)
Kp4=Kc_*(1+Td_/Ti_);Ti4=Ti_+Td_;Td4=Ti_*Td_/Ti4;print("Kp",Kp4,"1+",1+Td_/Ti_,"Ti",Ti4,"Td",Td4,"Ki",Kp4/Ti4,"Kd",Kp4*Td4,"HF",Kp4*(1+1/.1))
p4=(Kp4,Ti4,Td4);print("margins 4.1",mar(L(p4,Gfull)),mar(L(p4,Gred)))
t,y,u,den=cl(*p4,GFu,10,12,.397146,tend=600);print("4.1 full",st(t,y,u,10,12),"umin",u.min(),"peaks");
from scipy.signal import find_peaks
pk,_=find_peaks(y-12);print([(round(t[k],0),round(y[k]-12,3)) for k in pk[:3]]);print("poles full cl max re",max(np.roots(den).real))
t,y2,u2,den2=cl(*p4,GFo,10,12,K3,tend=250);print("FOPDT cl poles",np.roots(den2),"max re",max(np.roots(den2).real))
fig,ax=plt.subplots(1,2,figsize=(9,3));ax[0].plot(t,np.where(t>=50,12,10),'k:');ax[0].plot(t,y,'k') if len(t)==len(y) else None
t6,y6,u6,_=cl(*p4,GFu,10,12,.397146,tend=600);ax[0].cla();ax[0].plot(t6,np.where(t6>=50,12,10),'k:',label='SP');ax[0].plot(t6,y6,'k',label='FT completa');ax[1].plot(t6,u6,'k');ax[1].axhline(0,color='gray',ls=':');ax[1].axhline(100,color='gray',ls=':');ax[0].legend();ax[0].set_title('VP');ax[1].set_title('SC');plt.tight_layout();plt.savefig(F+'i41.png');plt.close()
t,yq,uq,_=cl(*p4,GFo,10,12,K3,tend=250);plt.figure(figsize=(4.5,2.8));plt.plot(t,yq,'r--',label='FOPDT (instável)');plt.plot(t6[t6<250],y6[t6<250],'k',label='completa');plt.ylim(-20,40);plt.legend();plt.xlabel('t [s]');plt.tight_layout();plt.savefig(F+'i41_unst.png');plt.close()
# P-only test
fig,ax=plt.subplots(1,3,figsize=(9,2.6))
for a,k in zip(ax,(.5,1,1.2)):
    Kc=k*Kcu;den=padd(FT[2][1],Kc*FT[2][0]);t=np.arange(0,500,.1);_,y,_=lsim(TF(Kc*FT[2][0],den),np.ones_like(t)*1.0,t);a.plot(t,y,'k');a.set_title('Kc=%.1f'%Kc);a.set_xlabel('t [s]')
plt.tight_layout();plt.savefig(F+'i41_kcu.png');plt.close()
# Routh check at Kc: poles of 1+Kc G
for k in (.5,1,1.2):
    Kc=k*Kcu;print(k,np.roots(padd(FT[2][1],Kc*FT[2][0])))
