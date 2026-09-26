"""Confidence intervals, figures, lower bounds and summary table."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import lower_bound_analysis as lb
ROOT=Path(__file__).resolve().parent
P=['Early','Intermediate','Latest','Combined'];COLORS=['#007BFF','#FF7A00','#009E45','#222222'];STYLES=['-','--','-.',':']

def intervals():
    meta=np.load(ROOT/'matrix_meta.npy',mmap_mode='r');hits=np.load(ROOT/'full_hits.npy',mmap_mode='r');jid=np.load(ROOT/'matrix_journal_ids.npy',mmap_mode='r')
    J=len(json.loads((ROOT/'journal_ids.json').read_text()));z=np.load(ROOT/'Full_counts.npz');months=z['months'];n=z['N'];h=z['H'];base=(months>=201801)&(months<=202112)
    weights=n[base].sum(0)/n[base].sum();influence=np.zeros((len(months),J,4));cluster_n=np.zeros(len(months),int)
    for m,month in enumerate(months):
        row=np.flatnonzero(meta[:,1]==month);b=np.minimum(meta[row,3]//50,10);nn=np.bincount(b,minlength=11);assert np.array_equal(nn,n[m]);jj=jid[row];cluster_n[m]=len(np.unique(jj))
        for p in range(4):
            prob=h[m,:,p]/nn;contribution=weights[b]*(hits[row,p]-prob[b])/nn[b]
            influence[m,:,p]=np.bincount(jj,weights=contribution,minlength=J)
    row=np.flatnonzero((meta[:,1]>=201801)&(meta[:,1]<=202112));b=np.minimum(meta[row,3]//50,10);nn=n[base].sum(0);jj=jid[row];ref=np.zeros((J,4))
    for p in range(4):
        prob=h[base,:,p].sum(0)/nn;contribution=weights[b]*(hits[row,p]-prob[b])/nn[b]
        ref[:,p]=np.bincount(jj,weights=contribution,minlength=J)
    delta=influence-ref;se=np.sqrt((influence**2).sum(1)*cluster_n[:,None]/(cluster_n[:,None]-1));dse=np.sqrt((delta**2).sum(1)*J/(J-1))
    f=pd.read_csv(ROOT/'full_monthly.csv');r=[]
    for m,month in enumerate(months):
        for p,panel in enumerate(P):r.append(dict(month=int(month),panel=panel,se_percent=100*se[m,p],excess_se_pp=100*dse[m,p]))
    f=f.merge(pd.DataFrame(r),on=['month','panel']);f['ci_low']=(f.observed_percent-1.96*f.se_percent).clip(0,100);f['ci_high']=(f.observed_percent+1.96*f.se_percent).clip(0,100)
    f['excess_ci_low']=f.excess_pp-1.96*f.excess_se_pp;f['excess_ci_high']=f.excess_pp+1.96*f.excess_se_pp
    f.to_csv(ROOT/'Monthly_intervals.csv',index=False)
    rows=[]
    for label,take in [('2022',months//100==2022),('2023_to_Aug2026',months>=202301),('August2026',months==202608)]:
        ss=np.sqrt((delta[take].mean(0)**2).sum(0)*J/(J-1))
        for p,panel in enumerate(P):
            value=f[(f.panel==panel)&f.month.isin(months[take])].excess_pp.mean();rows.append(dict(period=label,panel=panel,excess_pp=value,ci_low=value-196*ss[p],ci_high=value+196*ss[p]))
    pd.DataFrame(rows).to_csv(ROOT/'Period_intervals.csv',index=False)
    return f

def plot(frame,metric,label,stem,lo,hi,legend):
    plt.rcParams.update({'font.size':13,'font.family':'DejaVu Sans'})
    fig,ax=plt.subplots(figsize=((9.721518987 if metric=='conditional_bound_percent' else 10),5))
    for name,color,style in zip(P,COLORS,STYLES):
        a=frame[frame.panel==name];dates=pd.to_datetime(a.month.astype(str),format='%Y%m')
        ax.plot(dates,a[metric],color=color,ls=style,lw=2.8,label=f'{name}: {90 if name=="Combined" else 30} words');ax.fill_between(dates,a[lo],a[hi],color=color,alpha=.12,lw=0)
    ax.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2026-10-01'));ax.set_ylabel(label);ax.axvline(pd.Timestamp('2023-01-01'),color='#777777',lw=1,ls=':')
    ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));ax.grid(alpha=.16);ax.legend(ncol=2,loc=legend,frameon=False)
    if metric!='excess_pp':ax.set_ylim(0,100);ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    else:ax.axhline(0,color='#777777',lw=.8)
    fig.tight_layout();fig.savefig(ROOT/(stem+'.png'),dpi=500);fig.savefig(ROOT/(stem+'.svg'));plt.close(fig)

def ai_bounds():
    c=np.load(ROOT/'Matched_validation_counts.npz');months=c['months'];n=c['N'];h=np.stack([c['H'][1-s,s] for s in [0,1]])
    keys,counts,journals=lb.journal_counts(months);cell=keys>>32;jid=keys&((1<<32)-1);shape=(n.size,len(journals))
    nj=sparse.csr_matrix((counts[:,0],(cell,jid)),shape=shape);hj=[sparse.csr_matrix((counts[:,p+1],(cell,jid)),shape=shape) for p in range(4)]
    assert np.array_equal(np.asarray(nj.sum(1)).reshape(n.shape),n)
    for p in range(4):assert np.array_equal(np.asarray(hj[p].sum(1)).reshape(n.shape),h[:,:,:,p])
    frame,strata=lb.trend_bounds(months,n,h,nj,hj);frame.to_csv(ROOT/'AI_lower_bound_monthly.csv',index=False);pd.DataFrame(strata).to_csv(ROOT/'AI_lower_bound_length_strata.csv',index=False)
    summary=[]
    for p,g in frame.groupby('panel'):
        for label,take in [('Jan–Oct 2022 check',g.month.between(202201,202210)),('August 2026',g.month==202608)]:
            a=g[take];summary.append(dict(panel=p,period=label,n=int(a.n.sum()),conditional_bound_percent=np.average(a.conditional_bound_percent,weights=a.n)))
    pd.DataFrame(summary).to_csv(ROOT/'AI_lower_bound_summary.csv',index=False)
    (ROOT/'AI_lower_bound_checks.json').write_text(json.dumps(dict(all_eligible_records_scored=int(n.sum()),all_denominators_match=True,all_held_out_union_counts_match_independent_feature_pass=True,each_record_uses_opposite_half_vocabulary=True,reference_period='2018–2021',reference_model='2018–2021 binomial logit trend with month indicators',interpretation='conditional on the historical model representing unassisted writing'),indent=2))
    return frame

def plot_projection(frame=None):
    """Observed coverage (P) and projected unassisted coverage (Q) for each word set."""
    if frame is None:frame=pd.read_csv(ROOT/'AI_lower_bound_monthly.csv')
    style={'font.size':13,'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'svg.fonttype':'none'}
    with plt.rc_context(style):_projection_figure(frame)

def _projection_figure(frame):
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,name,color in zip(axs.ravel(),P,COLORS):
        a=frame[frame.panel==name];dates=pd.to_datetime(a.month.astype(str),format='%Y%m')
        ax.axvspan(pd.Timestamp('2018-01-01'),pd.Timestamp('2022-01-01'),color='#EDEDED',lw=0,zorder=0)
        ax.axvline(pd.Timestamp('2023-01-01'),color='#777777',lw=1,ls=':',zorder=1)
        ax.plot(dates,a.target_coverage_percent,color=color,lw=2.8,zorder=3,label='Observed')
        ax.plot(dates,a.reference_coverage_at_target_length_percent,color='#8C8C8C',lw=2.8,ls='--',zorder=2,label='Projected (unassisted)')
        ax.set_title(f'{name}: {90 if name=="Combined" else 30} words',loc='left',fontsize=14)
        ax.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2026-10-01'));ax.set_ylim(0,100)
        ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
        ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(alpha=.16);ax.legend(loc='upper left',frameon=False,fontsize=12)
    axs[0,0].text(pd.Timestamp('2020-01-01'),4,'Fitting period',ha='center',va='bottom',fontsize=11,color='#555555')
    for ax in axs[:,0]:ax.set_ylabel('Abstracts containing selected words (%)')
    fig.tight_layout();fig.savefig(ROOT/'Figure2_projected_coverage.png',dpi=500);fig.savefig(ROOT/'Figure2_projected_coverage.svg');plt.close(fig)

def report(f):
    """Write the summary table used in the paper."""
    sets=json.loads((ROOT/'full_sets.json').read_text());cv=pd.read_csv(ROOT/'Cross_validated_monthly.csv');folds=[json.loads((ROOT/f'train{i}_sets.json').read_text()) for i in [0,1]]
    rows=[]
    for p in P:
        g=f[f.panel==p];a=g[g.month==202608].iloc[0];c=cv[(cv.panel==p)&(cv.month==202608)].iloc[0];peak=g.loc[g.observed_percent.idxmax()]
        rows.append(dict(panel=p,reference_percent=a.baseline_percent,August2026_percent=a.observed_percent,August2026_excess_pp=a.excess_pp,August2026_cv_excess_pp=c.excess_pp,mean2022_excess_pp=g[g.month//100==2022].excess_pp.mean(),mean_post2022_excess_pp=g[g.month>=202301].excess_pp.mean(),peak_month=int(peak.month),peak_percent=peak.observed_percent,selected_in_both_training_fits=sum(w in folds[0][p] and w in folds[1][p] for w in sets[p])))
    s=pd.DataFrame(rows);s.to_csv(ROOT/'Summary.csv',index=False)
    print(s.to_string(index=False),flush=True)

def main():
    f=intervals();plot(f,'observed_percent','Abstracts containing selected words (%)','Figure1_coverage','ci_low','ci_high','lower left')
    b=ai_bounds();plot_projection(b);plot(b,'conditional_bound_percent','Conditional lower bound on AI assistance (%)','Figure3_lower_bound','ci_low_percent','ci_high_percent','upper left');report(f)
    import subprocess,sys
    subprocess.run([sys.executable,str(ROOT/'check_lower_bound_gradient.py')],check=True)

if __name__=='__main__':main()
