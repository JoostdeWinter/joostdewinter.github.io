"""Conditional lower bounds on the share of AI-assisted abstracts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit,logit
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parent
PANELS=['Early','Intermediate','Latest','Combined']
COLORS=['#007BFF','#FF7A00','#009E45','#222222']
STYLES=['-','--','-.',':']

def journal_counts(months):
    vocab=json.loads((ROOT/'candidates.json').read_text());vi={w:i for i,w in enumerate(vocab)}
    sets=[json.loads((ROOT/f'train{i}_sets.json').read_text()) for i in [0,1]]
    ids=[[np.array([vi[w] for w in ss[p]]) for p in PANELS] for ss in sets]
    # The cached journal aggregates are used only if this signature of the inputs matches.
    manifest=json.loads((ROOT/'Release_manifest.json').read_text())
    signature=hashlib.sha256(json.dumps({'sets':sets,'vocab':vocab,
        'features':{k:v for k,v in manifest['files'].items() if k.startswith('features/')}},sort_keys=True).encode()).hexdigest()
    cache=ROOT/'Lower_bound_journal_counts.npz'
    if cache.exists():
        c=np.load(cache)
        if str(c['signature'])==signature:
            return c['keys'],c['counts'],c['journals']
    lookup={};keys=[];values=[]
    for path in sorted((ROOT/'features').glob('*.npz')):
        with np.load(path) as a:
            bits=a['bits'];meta=a['meta'];names,inv=np.unique(a['journals'],return_inverse=True)
            codes=np.array([lookup.setdefault(str(s),len(lookup)) for s in names],np.int64)[inv]
            mi=np.searchsorted(months,meta[:,1]);assert np.array_equal(months[mi],meta[:,1])
            cell=(meta[:,2]*len(months)+mi)*11+np.minimum(meta[:,3]//50,10)
            key=(cell.astype(np.int64)<<32)|codes
            u,ix=np.unique(key,return_inverse=True)
            counts=np.zeros((len(u),5),np.int64);counts[:,0]=np.bincount(ix,minlength=len(u))
            for start in range(0,len(meta),100000):
                end=min(start+100000,len(meta));b=bits[start:end];fold=meta[start:end,2]
                for panel in range(4):
                    hit=np.zeros(len(b),bool)
                    for split in [0,1]:
                        take=fold==split;ii=ids[1-split][panel]
                        hit[take]=((b[take][:,ii//8]&(1<<(ii%8)).astype(np.uint8))!=0).any(axis=1)
                    counts[:,panel+1]+=np.bincount(ix[start:end][hit],minlength=len(u))
            keys.append(u);values.append(counts)
        print('Journal aggregates:',path.name,flush=True)
    k=np.concatenate(keys);v=np.concatenate(values);order=np.argsort(k)
    k=k[order];v=v[order];u,start=np.unique(k,return_index=True)
    v=np.add.reduceat(v,start,axis=0)
    journals=np.array(list(lookup))
    np.savez_compressed(cache,keys=u,counts=v,journals=journals,signature=signature)
    return u,v,journals

def logistic_fit(x,n,h):
    initial=np.zeros(x.shape[1]);initial[0]=logit((h.sum()+.5)/(n.sum()+1))
    def objective(beta):
        eta=x@beta
        return np.sum(n*np.logaddexp(0,eta)-h*eta)/n.sum(),x.T@(n*expit(eta)-h)/n.sum()
    opt=minimize(objective,initial,jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':1000})
    assert np.max(np.abs(opt.jac))<1e-7
    beta=opt.x
    for _ in range(8):
        pp=expit(x@beta);fisher=x.T@((n*pp*(1-pp))[:,None]*x)
        step=np.linalg.solve(fisher,x.T@(h-n*pp));beta+=step
        if np.max(np.abs(step))<1e-11:break
    pp=expit(x@beta);fisher=x.T@((n*pp*(1-pp))[:,None]*x)
    return beta,np.linalg.inv(fisher),pp

def trend_bounds(months,n,h,nj,hj):
    """Binomial logit trend plus calendar-month indicators, fitted 2018--2021.

    The time coefficient models drift. Eleven month indicators model seasonality.
    Every fit uses only the pre-AI reference records in the evaluation half.
    The vocabulary itself was selected in the opposite half.
    """
    M=len(months);time=months//100+(months%100-.5)/12-2020
    x=np.column_stack([np.ones(M),time]+[(months%100==k).astype(float) for k in range(2,13)])
    refs=np.flatnonzero((months>=201801)&(months<=202112));xx=x[refs]
    model={};parameters=[]
    for f in [0,1]:
        for b in range(11):
            for p in range(4):
                beta,inv,fit=logistic_fit(xx,n[f,refs,b],h[f,refs,b,p])
                model[f,b,p]=(expit(x@beta),inv,fit)
                parameters.append(dict(held_out_half=f,length_band=b,panel=PANELS[p],
                    intercept=beta[0],time_slope_per_year=beta[1],
                    **{f'month_{m}':beta[m] for m in range(2,13)}))
    pd.DataFrame(parameters).to_csv(ROOT/'AI_lower_bound_trend_coefficients.csv',index=False)
    rows=[];strata=[]
    for mi,month in enumerate(months):
        nt=n[:,mi];N=nt.sum();p=h[:,mi]/nt[:,:,None]
        for panel,name in enumerate(PANELS):
            q=np.array([[model[f,b,panel][0][mi] for b in range(11)] for f in [0,1]])
            pp=p[:,:,panel];bb=((pp-q)/(1-q)).clip(0,1);active=pp>q
            B=float((nt*bb).sum()/N);ch=np.zeros(n.size);cn=np.zeros(n.size)
            for f in [0,1]:
                for b in range(11):
                    cell=(f*M+mi)*11+b;dh=1/(1-q[f,b]) if active[f,b] else 0.
                    ch[cell]+=dh/N;cn[cell]+=(bb[f,b]-B-pp[f,b]*dh)/N
                    dq=(pp[f,b]-1)/(1-q[f,b])**2 if active[f,b] else 0.
                    prediction,inv,fit=model[f,b,panel]
                    coeff=nt[f,b]/N*dq*q[f,b]*(1-q[f,b])*(xx@inv@x[mi])
                    cells=(f*M+refs)*11+b
                    ch[cells]+=coeff;cn[cells]-=coeff*fit
                    strata.append(dict(reference='2018–2021 trend',month=int(month),panel=name,
                        held_out_half=f,length_band=b,target_n=int(nt[f,b]),
                        reference_n=int(n[f,refs,b].sum()),target_rate=pp[f,b],reference_rate=q[f,b],bound=bb[f,b]))
            influence=np.asarray(hj[panel].T@ch+nj.T@cn).ravel()
            assert abs(influence.sum())<1e-8
            used=np.flatnonzero((ch!=0)|(cn!=0));clusters=int(np.count_nonzero(np.asarray(nj[used].sum(0)).ravel())) if len(used) else 0
            se=np.sqrt(np.dot(influence,influence)*clusters/(clusters-1)) if clusters>1 else 0.
            rows.append(dict(reference='2018–2021 trend',month=int(month),panel=name,n=int(N),
                reference_n=int(n[:,refs].sum()),target_coverage_percent=100*(nt*pp).sum()/N,
                reference_coverage_at_target_length_percent=100*(nt*q).sum()/N,
                conditional_bound_percent=100*B,se_percent=100*se,
                ci_low_percent=100*max(0,B-1.96*se),ci_high_percent=100*min(1,B+1.96*se),
                one_sided_95_lower_limit_percent=100*max(0,B-1.645*se),journal_clusters=clusters))
    return pd.DataFrame(rows),pd.DataFrame(strata)
