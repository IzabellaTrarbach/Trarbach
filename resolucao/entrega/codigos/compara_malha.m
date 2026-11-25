% COMPARACAO BANCADA x SIMULACAO (FT completa do circuito RC) - estilo do script do professor
% ATENCAO: NAO foi executado no MATLAB (indisponivel onde foi escrito). A logica espelha simexp.py /
% simula_bancada.py, que foram testados com os CSVs reais Q11, Q12, Q13a e Q13b.
% Diferenca importante em relacao ao script do professor: o CSV real do ScadaBR tem as variaveis com numeros
% de linhas DIFERENTES (ex.: SO 2024, VP 2028), entao NAO se pode dividir por 5. Aqui as variaveis sao
% separadas pelo NOME do ponto e alinhadas numa grade de 1 s.
clear; clc;
% ---------- 1) Arquivo e trecho a comparar (EDITE) ----------
arquivo='Q11.csv';
inicio=1605;        % segundo (1 = 1a amostra) em que a malha foi fechada; SO(inicio) = SC manual imediatamente antes
% ---------- 2) Circuito (valores medidos) ----------
R=[22000 46800 32600 66800]; C=[947e-6 947e-6 955e-6]; g=1./R;
off=-0.262;         % offset da saida (VP = K*SO + off), do ajuste linear dos patamares
% ---------- 3) Controlador (EDITE) ----------
Kp=3.523; Ti=90.99; Td=11.114; mu=0.1;
alpha=0;            % 0 = PI_D; >0 = PI_D2GL
modo='PI_D';        % 'PI_D' ou 'I_PD'
umin=9.99485; umax=99.940875;   % limites do SC observados nos CSVs
aw='clamp';         % 'clamp' (anti-windup) ou 'none'
Rp=0; tRp=0;        % item 2.2: Rp [ohm] e instante (s, relativo ao inicio do trecho) em que foi ligado
% ---------- 4) Leitura (formato real do ScadaBR) ----------
T=readtable(arquivo,'Delimiter',',','Encoding','ISO-8859-1','VariableNamingRule','preserve','DecimalSeparator','.');
nomes=strtrim(string(T{:,1})); tempo=datetime(string(T{:,2}),'InputFormat','yyyy/MM/dd HH:mm:ss'); valor=T{:,3};
SO=serie(nomes,tempo,valor,'SO'); SP=serie(nomes,tempo,valor,'SP'); VP=serie(nomes,tempo,valor,'VP');
SO=SO(inicio:end); SP=SP(inicio:end); VP=VP(inicio:end); t=(0:length(SO)-1)';
% ---------- 5) Simulacao discreta (passo 0,05 s) ----------
A=[-(g(1)+g(2))/C(1) g(2)/C(1) 0; g(2)/C(2) -(g(2)+g(3))/C(2) g(3)/C(2); 0 g(3)/C(3) -(g(3)+g(4))/C(3)];
B=[g(1)/C(1);0;0]; dt=0.05; n=floor(length(SO)/dt);
gp=0; if Rp>0, gp=1/Rp; end
A2=A; A2(3,3)=A(3,3)-gp/C(3);
Phi=expm(A*dt); Gam=(A\(Phi-eye(3)))*B; Phi2=expm(A2*dt); Gam2=(A2\(Phi2-eye(3)))*B;
x=(-A\B)*SO(1); x=x*(VP(1)-off)/x(3);                 % estado de regime compativel com VP(1)
Ki=Kp/Ti; Kd=Kp*Td; Tf=mu*Td; y0=x(3)+off;
I=SO(1); if strcmp(modo,'I_PD'), I=SO(1)+Kp*y0; end
xf=y0; xs=SP(1); vps=zeros(n,1); scs=zeros(n,1); ts=(0:n-1)'*dt;
for k=1:n
  tk=ts(k); spk=SP(min(floor(tk)+1,length(SP)));
  if Rp>0 && tk>=tRp, Phi=Phi2; Gam=Gam2; end
  if alpha>0, xs=xs+(1-exp(-dt/Ti))*(spk-xs); spf=(1-alpha)*spk+alpha*xs; else, spf=spk; end
  y=x(3)+off; xf=xf+(1-exp(-dt/Tf))*(y-xf); der=Kd*(y-xf)/Tf; e=spf-y;
  if strcmp(modo,'PI_D'), u=Kp*e+I-der; else, u=I-Kp*y-der; end
  uc=min(max(u,umin),umax);
  if strcmp(aw,'clamp')
    if ~((u>umax && e>0) || (u<umin && e<0)), I=I+Ki*e*dt; end
  else, I=I+Ki*e*dt; end
  vps(k)=y; scs(k)=uc; x=Phi*x+Gam*uc;
end
% ---------- 6) Comparacao ----------
idx=min(round(t/dt)+1,n); vpm=vps(idx); scm=scs(idx); somed=movmedian(SO,15);
fprintf('VP: RMS %.2f %% | max %.2f %%\n',sqrt(mean((vpm-VP).^2)),max(abs(vpm-VP)));
fprintf('SC (SO com mediana movel 15 s): RMS %.2f %% | SC bruto: RMS %.2f %%\n',sqrt(mean((scm-somed).^2)),sqrt(mean((scm-SO).^2)));
figure; subplot(1,2,1); plot(t,SP,'k:',t,VP,'b',ts,vps,'r--'); grid; legend('SP','VP exp.','VP sim.'); xlabel('t [s]'); ylabel('%');
subplot(1,2,2); plot(t,SO,'Color',[.62 .79 .88]); hold on; plot(t,somed,'b',ts,scs,'r--'); grid; legend('SO bruto','SO mediana','SC sim.'); xlabel('t [s]');

function v=serie(nomes,tempo,valor,var)
  % serie de 1 s do ponto cujo nome termina em var (ex.: 'uno - SO'); repete o ultimo valor sem registro
  idx=endsWith(nomes,var); t0=min(tempo); n=round(seconds(max(tempo)-t0))+1;
  v=nan(n,1); v(round(seconds(tempo(idx)-t0))+1)=valor(idx);
  for i=2:n, if isnan(v(i)), v(i)=v(i-1); end, end
  v(isnan(v))=v(find(~isnan(v),1));
end
