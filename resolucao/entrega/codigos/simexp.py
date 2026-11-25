import numpy as np
from scipy.linalg import expm
R=np.array([22000,46800,32600,66800.]);C=np.array([947e-6,947e-6,955e-6]);g=1/R
A=np.array([[-(g[0]+g[1])/C[0], g[1]/C[0],0],[g[1]/C[1],-(g[1]+g[2])/C[1],g[2]/C[1]],[0,g[2]/C[2],-(g[2]+g[3])/C[2]]]);B=np.array([g[0]/C[0],0,0])
OFF=-0.262            # offset medido do VP (VP = K*SO + OFF), ver ajuste linear
SOMIN,SOMAX=9.99485,99.940875   # limites do SO observados nos 4 arquivos
def simula(SP,so0,vp0,Kp,Ti,Td,mu=0.1,dt=0.05,sat=True,aw='clamp',modo='PI_D',off=OFF,x0=None,pre=None):
    """SP: vetor 1 amostra/s a partir do instante em que a malha e fechada. so0: SC (manual) imediatamente antes.
    O estado do circuito e ajustado para que VP(0)=vp0 (estado estacionario compativel com so0)."""
    Ki,Kd=Kp/Ti,Kp*Td; Tf=mu*Td if Td>0 else 1.0
    Phi=expm(A*dt);Gam=np.linalg.solve(A,Phi-np.eye(3))@B
    x=np.linalg.solve(-A,B)*so0; x=x*(vp0-off)/x[2]          # escala para VP(0)=vp0
    if x0 is not None: x=np.array(x0,float)
    npre=0 if pre is None else len(pre)
    I=so0 if modo=='PI_D' else so0+Kp*vp0
    if pre is not None: I=pre[-1] if modo=='PI_D' else pre[-1]+Kp*vp0
    xf=vp0; n=int(len(SP)/dt);VP=np.zeros(n);SC=np.zeros(n)
    for k in range(n):
        sp=SP[min(int(k*dt),len(SP)-1)];y=x[2]+off
        if k*dt<npre:
            u=pre[int(k*dt)];VP[k]=y;SC[k]=u;x=Phi@x+Gam*u;xf=y;continue
        xf+=(1-np.exp(-dt/Tf))*(y-xf);der=Kd*(y-xf)/Tf if Td>0 else 0.0
        e=sp-y;u=(Kp*e+I-der) if modo=='PI_D' else (I-Kp*y-der)
        uc=min(max(u,SOMIN),SOMAX) if sat else u
        if aw=='clamp':
            if not ((u>SOMAX and e>0) or (u<SOMIN and e<0)): I+=Ki*e*dt
        else: I+=Ki*e*dt
        VP[k]=y;SC[k]=uc;x=Phi@x+Gam*uc
    t=np.arange(n)*dt;return t,VP,SC
def amostra(t,v,n): return np.interp(np.arange(n),t,v)
def mediana_movel(x,w=15):
    h=w//2;p=np.pad(x,(h,h),mode='edge');return np.array([np.median(p[i:i+w]) for i in range(len(x))])
