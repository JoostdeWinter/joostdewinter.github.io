"""Numerically check the fitted-reference influence used in uncertainty propagation."""
import numpy as np
from scipy.special import expit
from lower_bound_analysis import ROOT,logistic_fit
c=np.load(ROOT/'Matched_validation_counts.npz');n=c['N'];m=c['months'];h=np.stack([c['H'][1-s,s] for s in [0,1]])
j=np.load(ROOT/'Lower_bound_journal_counts.npz');cell=j['keys']>>32;jid=j['keys']&((1<<32)-1);cnt=j['counts']
base=np.flatnonzero((m>=201801)&(m<=202112));X=np.column_stack([np.ones(len(m)),m//100+(m%100-.5)/12-2020]+[(m%100==k).astype(float) for k in range(2,13)])
f=0;b=5;p=3;cells=(f*len(m)+base)*11+b;take=np.isin(cell,cells)
counts=np.bincount(jid[take],weights=cnt[take,0]);journal=int(np.argmax(counts));dn=np.zeros(len(base));dh=np.zeros(len(base));lookup={int(x):i for i,x in enumerate(cells)}
for cc,v in zip(cell[take&(jid==journal)],cnt[take&(jid==journal)]):
 dn[lookup[int(cc)]]+=v[0];dh[lookup[int(cc)]]+=v[p+1]
xx=X[base];nn=n[f,base,b].astype(float);hh=h[f,base,b,p].astype(float);beta,inv,fit=logistic_fit(xx,nn,hh)
t=int(np.flatnonzero(m==202608)[0]);q=expit(X[t]@beta);analytic=q*(1-q)*X[t]@inv@xx.T@(dh-fit*dn)
eps=.001
plus=logistic_fit(xx,nn+eps*dn,hh+eps*dh)[0];minus=logistic_fit(xx,nn-eps*dn,hh-eps*dh)[0]
numeric=(expit(X[t]@plus)-expit(X[t]@minus))/(2*eps)
assert np.isclose(numeric,analytic,rtol=1e-4,atol=1e-9),(numeric,analytic)
print({'analytic':float(analytic),'finite_difference':float(numeric),'absolute_error':float(abs(numeric-analytic))})
