"""Run the full analysis: screening, grouping, word selection, cross-validation, lower bound and figures."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import engine as e

ROOT=Path(__file__).resolve().parent
Z=np.load(ROOT/'inputs/monthly_length_word_counts.npz')
W=Z['W'];N=Z['N'];M=Z['months'];V=Z['vocabulary'];VI={w:i for i,w in enumerate(V)}
BASE=(M>=201801)&(M<=202112);POST=(M>=202301)&(M<=202608)
P=['Early','Intermediate','Latest','Combined']

def rates(folds,weights=None):
    w=W[folds].sum(0);n=N[folds].sum(0)
    if weights is None:weights=n[BASE].sum(0)/n[BASE].sum()
    adjusted=np.einsum('mbw,mb->mw',w,weights[None,:]/n,optimize=True)
    reference=np.einsum('bw,b->w',w[BASE].sum(0),weights/n[BASE].sum(0))
    counts=w.sum(1);reference_counts=counts[BASE].sum(0)
    delta=adjusted-reference
    ratio=np.divide(adjusted,reference,out=np.full_like(adjusted,np.nan),where=reference>0)
    absolute=(delta[POST]>=.001).any(0)
    relative=((ratio[POST]>=2)&(counts[POST]>=20)).any(0)&(reference_counts>=20)
    return dict(adjusted=adjusted,reference=reference,weights=weights,delta=delta,
        ratio=ratio,counts=counts,reference_counts=reference_counts,passes=absolute|relative,
        absolute=absolute,relative=relative)

def peak_rows(r,groups,tag):
    rows=[];ii=np.flatnonzero(POST)
    for group,names in groups.items():
        for word in names:
            j=VI[word];q=r['reference'][j]
            for kind in ['absolute','relative']:
                scores=r['delta'][ii,j] if kind=='absolute' else np.where((r['counts'][ii,j]>=20)&(r['reference_counts'][j]>=20),r['ratio'][ii,j],np.nan)
                if not np.isfinite(scores).any():continue
                t=ii[np.nanargmax(scores)]
                rows.append(dict(fit=tag,group=group,word=word,ranking=kind,month=int(M[t]),
                    observed_percent=100*r['adjusted'][t,j],baseline_percent=100*q,
                    excess_pp=100*r['delta'][t,j],relative_frequency=r['ratio'][t,j],
                    target_word_n=int(r['counts'][t,j]),reference_word_n=int(r['reference_counts'][j])))
    out=[]
    for (g,k),a in pd.DataFrame(rows).groupby(['group','ranking'],sort=False):
        a=a.sort_values(['excess_pp' if k=='absolute' else 'relative_frequency','word'],ascending=[False,True]);a['rank']=np.arange(1,len(a)+1);out.append(a)
    return pd.concat(out,ignore_index=True)

def evaluate(hits,meta,folds,weights,tag,sample):
    take=np.isin(meta[:,2],folds);mm=np.searchsorted(M,meta[:,1]);band=np.minimum(meta[:,3]//50,10)
    cell=mm*11+band
    n=np.bincount(cell[take],minlength=len(M)*11).reshape(len(M),11)
    assert np.array_equal(n,N[folds].sum(0))
    h=np.stack([np.bincount(cell[take],weights=hits[take,j],minlength=n.size).reshape(n.shape) for j in range(4)],axis=-1).astype(np.int64)
    f=np.einsum('mbp,mb->mp',h,weights[None,:]/n)
    q=np.einsum('bp,b->p',h[BASE].sum(0),weights/n[BASE].sum(0))
    rows=[]
    for i,m in enumerate(M):
        for j,p in enumerate(P):rows.append(dict(fit=tag,sample=sample,month=int(m),panel=p,n=int(n[i].sum()),marker_n=int(h[i,:,j].sum()),raw_percent=100*h[i,:,j].sum()/n[i].sum(),observed_percent=100*f[i,j],baseline_percent=100*q[j],excess_pp=100*(f[i,j]-q[j])))
    return pd.DataFrame(rows),n,h

def main():
    config=dict(reference_years=[2018,2019,2020,2021],reference='all 48 months pooled',length_standardization='pooled 2018–2021 length distribution within training sample',
        semantic_eligibility='fixed 4,742-word eligibility list',screen_vocabulary=22580,
        absolute_screen_pp=.1,relative_screen_ratio=2,minimum_target_month_count=20,minimum_reference_total_count=20,
        groups=3,kmeans_init='k-means++',kmeans_n_init=30,kmeans_algorithm='lloyd',curve_scaling='subtract 44-month mean, divide by Euclidean norm',
        cluster_naming='ascending mean month index weighted by positive centroid values',words_per_group=30,selection_objective='mean of 44 monthly coverage differences from pooled 2018–2021',
        validation='two-way PMID split; screening, grouping, word selection and peak months determined in each training half and evaluated in the held-out half',random_seed=20260920)
    (ROOT/'config.json').write_text(json.dumps(config,indent=2))
    rr={};groups={};screen=[];members=[];checks=[];ranked=[]
    for tag,folds in [('train0',[0]),('train1',[1]),('full',[0,1])]:
        r=rates(folds);rr[tag]=r
        np.savez_compressed(ROOT/f'{tag}_word_rates.npz',**{k:r[k] for k in ['adjusted','reference','weights','delta','ratio','passes']})
        groups[tag]=e.discover(tag)
        for i in np.flatnonzero(r['passes']):screen.append(dict(fit=tag,word=str(V[i]),absolute=bool(r['absolute'][i]),relative=bool(r['relative'][i]),on_eligibility_list=str(V[i]) in e.WI))
        members.extend(dict(fit=tag,group=g,word=w) for g,ws in groups[tag].items() for w in ws)
        peaks=peak_rows(r,groups[tag],tag);ranked.append(peaks)
        if tag!='full':
            test=rates([1-folds[0]],r['weights'])
            for row in peaks[peaks['rank']<=10].to_dict('records'):
                j=VI[row['word']];t=int(np.flatnonzero(M==row['month'])[0])
                row.update(held_out_observed_percent=100*test['adjusted'][t,j],held_out_baseline_percent=100*test['reference'][j],held_out_excess_pp=100*test['delta'][t,j],held_out_relative_frequency=test['ratio'][t,j]);checks.append(row)
        print('Discovery',tag,{g:len(ws) for g,ws in groups[tag].items()},flush=True)
    pd.DataFrame(screen).to_csv(ROOT/'Screened_words.csv',index=False)
    pd.DataFrame(members).to_csv(ROOT/'Word_groups.csv',index=False)
    ranks=pd.concat(ranked,ignore_index=True);ranks.to_csv(ROOT/'All_fit_peak_rankings.csv',index=False)
    ranks[ranks.fit=='full'].to_csv(ROOT/'Peak_word_rankings.csv',index=False)
    ranks[(ranks.fit=='full')&(ranks['rank']<=10)].to_csv(ROOT/'Peak_top_words.csv',index=False)
    pd.DataFrame(checks).to_csv(ROOT/'Peak_word_validation.csv',index=False)
    bits,meta,journals=e.load_features();frames=[];H=np.zeros((2,2,len(M),11,4),np.int64)
    for tag,folds in [('train0',[0]),('train1',[1]),('full',[0,1])]:
        sets=e.select_words(tag,groups[tag],bits,meta,folds)
        hits=e.count_hits(bits,sets)
        if tag=='full':
            frame,n,h=evaluate(hits,meta,[0,1],rr[tag]['weights'],tag,'full');np.save(ROOT/'full_hits.npy',hits);frames.append(frame)
            frame.to_csv(ROOT/'full_monthly.csv',index=False)
            np.savez_compressed(ROOT/'Full_counts.npz',N=n,H=h,months=M)
        else:
            fit=folds[0]
            for split in [0,1]:
                sample='training' if split==fit else 'held_out'
                frame,n,h=evaluate(hits,meta,[split],rr[tag]['weights'],tag,sample);H[fit,split]=h;frames.append(frame)
                if sample=='held_out':frame.to_csv(ROOT/f'{tag}_monthly.csv',index=False)
        path=pd.read_csv(ROOT/f'{tag}_selection.csv')
        trainframe=next(a for a in frames[::-1] if a.fit.iloc[0]==tag and a['sample'].iloc[0] in ['training','full'])
        for p in P[:3]:assert abs(trainframe[(trainframe.panel==p)&(trainframe.month>=202301)].excess_pp.mean()-path[path.group==p].iloc[-1].excess_pp)<1e-9
        # Check all selected individual words across every month and both halves.
        for word in sets['Combined']:
            j=e.WI[word];hit=(bits[:,j//8]&(1<<(j%8)))!=0
            cell=(meta[:,2]*len(M)+np.searchsorted(M,meta[:,1]))*11+np.minimum(meta[:,3]//50,10)
            counted=np.bincount(cell[hit],minlength=N.size).reshape(N.shape)
            assert np.array_equal(counted,W[:,:,:,VI[word]])
        print('Evaluated and checked',tag,flush=True)
    np.savez_compressed(ROOT/'Matched_validation_counts.npz',N=N,H=H,months=M)
    f=pd.concat(frames,ignore_index=True);f.to_csv(ROOT/'All_fit_monthly.csv',index=False)
    matched=[];cv=[]
    for (month,panel),a in f[f.fit!='full'].groupby(['month','panel']):
        held=a[a['sample']=='held_out'].sort_values('fit');train=a[a['sample']=='training'].sort_values('fit');weights=held.n
        row=dict(month=int(month),panel=panel,n=int(held.n.sum()),marker_n=int(held.marker_n.sum()))
        row.update({col:float(np.average(held[col],weights=weights)) for col in ['raw_percent','observed_percent','baseline_percent','excess_pp']});cv.append(row)
        tr=float(np.average(train.excess_pp,weights=weights));te=float(np.average(held.excess_pp,weights=weights))
        matched.append(dict(month=int(month),panel=panel,training_excess_pp=tr,held_out_excess_pp=te,training_minus_held_out_pp=tr-te))
    pd.DataFrame(cv).to_csv(ROOT/'Cross_validated_monthly.csv',index=False)
    a=pd.DataFrame(matched);a.to_csv(ROOT/'Matched_validation_monthly.csv',index=False)
    sm=[]
    for label,mask in [('January2023_August2026',a.month>=202301),('August2026',a.month==202608)]:
        for p,g in a[mask].groupby('panel'):sm.append(dict(period=label,panel=p,**{c:g[c].mean() for c in ['training_excess_pp','held_out_excess_pp','training_minus_held_out_pp']}))
    pd.DataFrame(sm).to_csv(ROOT/'Matched_validation_summary.csv',index=False)
    full=json.loads((ROOT/'full_sets.json').read_text());folds=[json.loads((ROOT/f'train{i}_sets.json').read_text()) for i in [0,1]]
    pd.DataFrame([dict(group=g,order=i+1,word=w,selected_in_both_folds=all(w in s[g] for s in folds)) for g in P[:3] for i,w in enumerate(full[g])]).to_csv(ROOT/'Thirty_words_per_phase.csv',index=False)
    (ROOT/'Validation.json').write_text(json.dumps(dict(abstracts=len(meta),reference_abstracts=int(N[:,BASE].sum()),all_month_fold_length_denominators_exact=True,selected_individual_counts_exact_all_months_and_both_halves=True,selection_objectives_match_direct_union_counts=True,screening_grouping_selection_in_each_training_half=True,eligibility_list_fixed=True,held_out_word_month_choices=len(checks),held_out_positive_choices=int((pd.DataFrame(checks).held_out_excess_pp>0).sum()),group_sizes=[30,30,30]),indent=2))
    del bits
    (ROOT/'matrix_bits.npy').unlink()
    print('Analysis complete',flush=True)
    import pooled_outputs
    pooled_outputs.main()

if __name__=='__main__':main()
