% COMPARACAO BANCADA x SIMULACAO (FT completa do circuito RC) - estilo do script do professor
% NAO foi executado no MATLAB (indisponivel onde foi escrito); a logica espelha simula_bancada.py (testado).
% Teste primeiro com o caso do item 1.1 (pico de SC da simulacao = 88 %).
clear; clc;
% ---------- 1) Dados medidos do circuito (EDITE) ----------
R=[22000 46800 32600 66800]; C=[947e-6 947e-6 955e-6]; g=1./R;
% ---------- 2) Parametros do PID (EDITE) ----------
Kp=3.523; Ti=90.99; Td=11.114; mu=0.1;
alpha=0;            % 0 = PI_D comum; >0 = PI_D2GL (item 2.1)
modo='PI_D';        % 'PI_D' ou 'I_PD' (item 4.2)
sat=false;          % true = satura SC em 0-100 (use em 1.3, 4.1 na bancada)
aw='none';          % 'none' | 'clamp' (anti reset windup)
Rp=0; tRp=0;        % item 2.2: Rp em ohm (0 = sem Rp) e instante (s, contado do inicio do trecho) em que foi ligado
inicio=1;           % amostra onde comeca o trecho a comparar
% ---------- 3) Leitura (igual ao script do professor) ----------
tablex=readtable('MALHAFECHADA.csv'); cellx=table2cell(tablex);
matx=cell2mat(cellx(:,3)); div=size(matx,1)/5;
SO=matx(1:div); SP=matx(div+1:2*div); VP=matx(2*div+1:3*div);   % SO = SC ; VP = Vc3
SO=SO(inicio:end); SP=SP(inicio:end); VP=VP(inicio:end); t=(0:length(SO)-1)';
% ---------- 4) Simulacao (discreta, passo 0,05 s) ----------
A=[-(g(1)+g(2))/C(1) g(2)/C(1) 0; g(2)/C(2) -(g(2)+g(3))/C(2) g(3)/C(2); 0 g(3)/C(3) -(g(3)+g(4))/C(3)];
B=[g(1)/C(1);0;0]; dt=0.05; n=floor(length(SO)/dt);
gp=0; if Rp>0, gp=1/Rp; end
A2=A; A2(3,3)=A(3,3)-gp/C(3);
Phi=expm(A*dt); Gam=(A\(Phi-eye(3)))*B; Phi2=expm(A2*dt); Gam2=(A2\(Phi2-eye(3)))*B;
x=(-A\B)*SO(1); Ki=Kp/Ti; Kd=Kp*Td; Tf=mu*Td;
y0=x(3); I=SO(1); if strcmp(modo,'I_PD'), I=SO(1)+Kp*y0; end
xf=y0; xs=SP(1); vps=zeros(n,1); scs=zeros(n,1); ts=(0:n-1)'*dt;
for k=1:n
  tk=ts(k); spk=SP(min(floor(tk)+1,length(SP)));
  if Rp>0 && tk>=tRp, Phi=Phi2; Gam=Gam2; end
  if alpha>0, xs=xs+(1-exp(-dt/Ti))*(spk-xs); spf=(1-alpha)*spk+alpha*xs; else spf=spk; end
  y=x(3); xf=xf+(1-exp(-dt/Tf))*(y-xf); der=Kd*(y-xf)/Tf; e=spf-y;
  if strcmp(modo,'PI_D'), u=Kp*e+I-der; else u=I-Kp*y-der; end
  uc=u; if sat, uc=min(max(u,0),100); end
  if strcmp(aw,'clamp')
    if ~((u>100 && e>0) || (u<0 && e<0)), I=I+Ki*e*dt; end
  else, I=I+Ki*e*dt; end
  vps(k)=y; scs(k)=uc; x=Phi*x+Gam*uc;
end
% ---------- 5) Comparacao ----------
idx=min(round(t/dt)+1,n); vpm=vps(idx); scm=scs(idx);
fprintf('VP: RMS %.2f %% | max %.2f %%\n',sqrt(mean((vpm-VP).^2)),max(abs(vpm-VP)));
fprintf('SC: RMS %.2f %% | max %.2f %%\n',sqrt(mean((scm-SO).^2)),max(abs(scm-SO)));
figure; subplot(1,2,1); plot(t,SP,'k:',t,VP,'b',ts,vps,'r--'); grid; legend('SP','VP exp.','VP sim.'); xlabel('t [s]'); ylabel('%');
subplot(1,2,2); plot(t,SO,'b',ts,scs,'r--'); grid; legend('SC exp. (SO)','SC sim.'); xlabel('t [s]');
