"""Select equal-size word sets and evaluate both reference and later rates out of sample."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import gc,json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
ROOT=Path(__file__).resolve().parent
GROUPS=['Early','Intermediate','Latest'];PANELS=GROUPS+['Combined']
WORDS=json.loads((ROOT/'candidates.json').read_text());WI={w:i for i,w in enumerate(WORDS)}
Z=np.load(ROOT/'inputs/monthly_length_word_counts.npz')
MONTHS=Z['months'];BASE=(MONTHS>=201801)&(MONTHS<=202112);POST=MONTHS>=202301
MI={int(m):i for i,m in enumerate(MONTHS)}
VOCAB=Z['vocabulary'];VI={w:i for i,w in enumerate(VOCAB)}

def discover(tag):
    r=np.load(ROOT/f'{tag}_word_rates.npz')
    ids=np.asarray([VI[w] for w in WORDS if r['passes'][VI[w]]])
    curves=r['delta'][POST][:,ids].T.copy()
    curves-=curves.mean(axis=1,keepdims=True)
    curves/=np.linalg.norm(curves,axis=1,keepdims=True)
    model=KMeans(n_clusters=3,init='k-means++',n_init=30,algorithm='lloyd',random_state=20260920).fit(curves)
    positive=np.maximum(model.cluster_centers_,0)
    timing=(positive@np.arange(curves.shape[1]))/positive.sum(axis=1)
    order=np.argsort(timing)
    pd.DataFrame([dict(fit=tag,group=name,cluster_id=int(k),word_count=int((model.labels_==k).sum()),
        positive_centroid_weighted_month_index=float(timing[k]),peak_month=int(MONTHS[POST][np.argmax(model.cluster_centers_[k])]))
        for name,k in zip(GROUPS,order)]).to_csv(ROOT/f'{tag}_grouping_diagnostics.csv',index=False)
    pd.DataFrame({name:model.cluster_centers_[k] for name,k in zip(GROUPS,order)},index=MONTHS[POST]).rename_axis('month').to_csv(ROOT/f'{tag}_centroids.csv')
    return {name:VOCAB[ids[model.labels_==k]].tolist() for name,k in zip(GROUPS,order)}

def load_features():
    paths=sorted((ROOT/'features').glob('*.npz'))
    n=json.loads((ROOT/'Eligibility_summary.json').read_text())['retained_records']
    width=(len(WORDS)+7)//8
    destination=ROOT/'matrix_bits.npy'
    bits=np.lib.format.open_memmap(destination,mode='w+',dtype=np.uint8,shape=(n,width))
    meta=np.empty((n,5),np.int32);journal_codes=np.empty(n,np.int32);journal_map={};start=0
    for path in paths:
        with np.load(path) as f:
            x=f['bits'];end=start+len(x);bits[start:end]=x
            meta[start:end]=f['meta']
            names,inverse=np.unique(f['journals'],return_inverse=True)
            codes=np.array([journal_map.setdefault(str(name),len(journal_map)) for name in names],np.int32)
            journal_codes[start:end]=codes[inverse];start=end
    bits.flush();assert start==n
    assert len(np.unique(meta[:,0]))==n
    np.save(ROOT/'matrix_meta.npy',meta);np.save(ROOT/'matrix_journal_ids.npy',journal_codes)
    (ROOT/'journal_ids.json').write_text(json.dumps(journal_map))
    return bits,meta,journal_codes

def segments(meta,folds):
    train=np.isin(meta[:,2],folds)
    mm=np.array([MI[int(m)] for m in meta[:,1]],dtype=np.int16)
    bands=np.minimum(meta[:,3]//50,10).astype(np.int8)
    take=train&(((meta[:,1]>=201801)&(meta[:,1]<=202112))|(meta[:,1]>=202301))
    row=np.flatnonzero(take);sid=mm[row]*11+bands[row]
    order=np.argsort(sid,kind='stable');row=row[order];sid=sid[order]
    unique,starts,counts=np.unique(sid,return_index=True,return_counts=True)
    assert len(unique)==(BASE.sum()+POST.sum())*11
    n=Z['N'][folds].sum(axis=0)
    length_weights=n[BASE].sum(axis=0)/n[BASE].sum()
    byte_counts=(counts+7)//8
    byte_starts=np.r_[0,np.cumsum(byte_counts)[:-1]]
    weights=[]
    for s,count in zip(unique,counts):
        m,b=divmod(int(s),11)
        if POST[m]:weight=length_weights[b]/count/POST.sum()
        else:
            weight=-length_weights[b]/n[BASE,b].sum()
        weights.append(weight)
    return row,starts,counts,byte_starts,byte_counts,np.array(weights)

def pack_group(bits,word_ids,segment):
    row,starts,counts,bs,bc,weights=segment
    packed=np.zeros((len(word_ids),int(bc.sum())),np.uint8)
    byte=np.asarray(word_ids)//8;mask=(1<<(np.asarray(word_ids)%8)).astype(np.uint8)
    for start,n,bstart in zip(starts,counts,bs):
        for offset in range(0,int(n),16000):
            rows=row[start+offset:start+min(offset+16000,int(n))]
            x=(bits[rows[:,None],byte[None,:]]&mask[None,:])!=0
            p=np.packbits(x.T,axis=1,bitorder='little')
            out_start=int(bstart)+offset//8
            packed[:,out_start:out_start+p.shape[1]]=p
    return packed

def select_words(tag,groups,bits,meta,folds):
    seg=segments(meta,folds);starts=seg[3];weights=seg[-1]
    sets={};path=[]
    for group in GROUPS:
        names=groups[group];ids=[WI[w] for w in names]
        assert len(ids)>=30
        packed=pack_group(bits,ids,seg)
        covered=np.zeros(packed.shape[1],np.uint8)
        totals=np.zeros(len(weights),np.uint32);chosen=[]
        for step in range(30):
            # Abstracts that the candidate word adds to the set, per month and length band; padded bits are zero.
            new=np.bitwise_count(packed&~covered)
            added=np.add.reduceat(new,starts,axis=1,dtype=np.uint32)
            del new
            gains=added@weights;gains[chosen]=-np.inf
            pick=int(np.argmax(gains));chosen.append(pick)
            totals+=added[pick];covered|=packed[pick]
            observed=totals[weights>0]@weights[weights>0]
            reference=-(totals[weights<0]@weights[weights<0])
            path.append(dict(fit=tag,group=group,step=step+1,word=names[pick],
                observed_percent=100*observed,baseline_percent=100*reference,
                excess_pp=100*(observed-reference),increment_pp=100*gains[pick]))
        sets[group]=[names[i] for i in chosen]
        print('Selected',tag,group,sets[group][:7],flush=True)
        del packed;gc.collect()
    sets['Combined']=sum([sets[g] for g in GROUPS],[])
    assert len(set(sets['Combined']))==90
    (ROOT/f'{tag}_sets.json').write_text(json.dumps(sets,indent=2))
    pd.DataFrame(path).to_csv(ROOT/f'{tag}_selection.csv',index=False)
    return sets

def count_hits(bits,sets):
    hits=np.empty((len(bits),4),np.uint8)
    for start in range(0,len(bits),100000):
        x=bits[start:start+100000]
        for g,name in enumerate(PANELS):
            ids=np.array([WI[w] for w in sets[name]])
            hits[start:start+len(x),g]=((x[:,ids//8]&(1<<(ids%8)).astype(np.uint8))!=0).any(axis=1)
    return hits
