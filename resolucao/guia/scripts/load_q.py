import numpy as np, datetime as dt
def load(nome):
    d={}
    for ln in open(nome,encoding='latin-1'):
        p=ln.strip().split(',')
        if len(p)<3 or not p[0].startswith('uno'): continue
        k=p[0].split('-')[1].strip()
        t=dt.datetime.strptime(p[1],'%Y/%m/%d %H:%M:%S');d.setdefault(k,[]).append((t,float(p[2])))
    t0=min(v[0][0] for v in d.values());t1=max(v[-1][0] for v in d.values())
    n=int((t1-t0).total_seconds())+1;out={'t0':t0}
    for k,v in d.items():
        a=np.full(n,np.nan)
        for t,x in v: a[int((t-t0).total_seconds())]=x
        # fill gaps with previous value
        for i in range(1,n):
            if np.isnan(a[i]): a[i]=a[i-1]
        if np.isnan(a[0]): a[0]=a[~np.isnan(a)][0]
        out[k]=a
    out['t']=np.arange(n)
    return out
