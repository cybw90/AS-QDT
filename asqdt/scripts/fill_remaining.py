#!/usr/bin/env python3
"""Fill the remaining table cells the shipped scripts didn't compute:
  (1) extended surrogate-ablation metrics: rel-Frobenius, per-family dMMD, ranking Spearman
  (2) FC forecaster baseline (held-out F1) per family
  (3) bridge sweep: within-distribution correction term + Delta_S + AUC over 5 seeds
Uses the cached mp_cache.pkl. Run: python -m asqdt.scripts.fill_remaining
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
from sklearn.metrics import f1_score, roc_auc_score

from asqdt import config as cfg
from asqdt.scripts.run_v2_revision import load_or_build_mp
from asqdt.scripts.run_v2_honest import fresh_noms, ensure_tank_stats, kscore
from asqdt.evaluation.fastq import gram_from_sv, mmd_u, batch_statevectors
from asqdt.quantum.surrogate_variants import SurrogateVariant
from asqdt.cpms.plant import CPMSPlant, ObserverDT
from asqdt.cpms.data import run_episode
from asqdt.cpms import cstr as C, tank as T

LAB={'fdi':'FDI','command':'Sinusoidal','ramp':'Ramp','composite':'Composite',
'cstr_fdi':'FDI','cstr_thermal':'Thermal','cstr_concentration':'Conc.',
'tank_bias':'Bias','tank_oscillation':'Oscill.','tank_drift':'Drift'}


def rel_frob(Ke, Ks):
    return np.linalg.norm(Ke-Ks,'fro')/(np.linalg.norm(Ke,'fro')+1e-12)


def ext_surrogate_metrics(MP):
    print("\n=== (1) Extended surrogate-ablation metrics (CNC, 5 seeds) ===")
    variants=['V0','V1','V2','V3','V4']
    acc={v:{'relf':[], 'dmmd':[], 'rho':[]} for v in variants}
    for seed in cfg.SEEDS:
        mp=MP['CNC'][seed]
        Xn_p,Xa_pool,qc=mp['Xn_p'],mp['Xa_pool'],mp['qcache']
        train_X=np.vstack([Xn_p,Xa_pool[:len(Xn_p)]]); K_train=qc.gram(train_X)
        full_X=np.vstack([Xn_p,Xa_pool]); K_full=qc.gram(full_X)
        tp=np.vstack([Xn_p[:6],Xa_pool[:6]]); K_tp=qc.gram(tp)
        vs={
         'V0':SurrogateVariant(15,(64,32,16),500,normalized=False,seed=0).fit(train_X,K_train),
         'V1':SurrogateVariant(15,(64,32,16),500,normalized=True,seed=0).fit(train_X,K_train),
         'V2':SurrogateVariant(64,(128,64),500,normalized=False,seed=0).fit(train_X,K_train),
         'V4':SurrogateVariant(15,(64,32,16),500,normalized=True,seed=0).fit(full_X,K_full)}
        v3=SurrogateVariant(15,(64,32,16),500,normalized=True,seed=0).fit(train_X,K_train)
        v3.refine_head_cka(train_X,K_train,maxiter=60,seed=0); vs['V3']=v3
        # exact per-family batch MMD (from mp['fam'])
        exact_mmd={a:mp['fam'][a]['q']['mmd'] for a in mp['data']['atk']}
        exact_order=[exact_mmd[a] for a in mp['data']['atk']]
        for vn,var in vs.items():
            acc[vn]['relf'].append(rel_frob(K_tp, var.kernel(tp)))
            dmmds=[]; sur_order=[]
            Kref=var.kernel(Xn_p)
            for a in mp['data']['atk']:
                Xa=(np.nan_to_num(mp['data']['atk'][a]['windows'])-mp['mu'])@mp['V']
                sm=mmd_u(Kref, var.kernel(Xa), var.kernel(Xn_p,Xa))
                dmmds.append(abs(sm-exact_mmd[a])); sur_order.append(sm)
            acc[vn]['dmmd'].append(np.mean(dmmds))
            acc[vn]['rho'].append(spearmanr(exact_order,sur_order).correlation)
    names={'V0':'MLP $D_e{=}15$','V1':'Normalized','V2':'$D_e{=}64$','V3':'Align-CKA','V4':'Expanded'}
    out={}
    for vn in variants:
        r=lambda k:(float(np.nanmean(acc[vn][k])),float(np.nanstd(acc[vn][k])))
        out[vn]=dict(relf=r('relf'),dmmd=r('dmmd'),rho=r('rho'))
        print(f"  {names[vn]:12s} relF={out[vn]['relf'][0]:.2f}±{out[vn]['relf'][1]:.2f}  "
              f"dMMD={out[vn]['dmmd'][0]:.3f}±{out[vn]['dmmd'][1]:.3f}  "
              f"rho_rank={out[vn]['rho'][0]:+.2f}")
    return out


def fc_baseline(MP):
    print("\n=== (2) FC forecaster baseline (held-out F1, 5 seeds) ===")
    out={}
    for bname in ('CNC','CSTR','Tank'):
        fams=list(MP[bname][cfg.SEEDS[0]]['data']['atk'].keys())
        acc={a:[] for a in fams}
        for seed in cfg.SEEDS:
            mp=MP[bname][seed]; d=mp['data']
            # Build forecaster on nominal reference RAW residual trajectories.
            # Reconstruct nominal residual trajectories (pre-window) for training.
            if bname=='CNC':
                p_out=6
                def nom_traj(ep):
                    p=CPMSPlant(seed=seed+100+ep); t=ObserverDT(0.05,seed=seed+1)
                    return run_episode(p,t,p.reference(cfg.HORIZON),sensor_mask=np.ones(6))['residuals']
                def atk_traj_list(a):
                    from asqdt.cpms.plant import ATTACK_GENERATORS
                    res=[]
                    for ep in range(10):
                        p=CPMSPlant(seed=seed+200+ep); t=ObserverDT(0.05,seed=seed+1)
                        ref=p.reference(cfg.HORIZON)
                        aq=ATTACK_GENERATORS[a](cfg.HORIZON,np.random.RandomState(seed+300+ep),E_max=2.0)
                        res.append(run_episode(p,t,ref,aq,np.ones(6))['residuals'])
                    return res
                Hh=cfg.HORIZON; W_L=10
            elif bname=='CSTR':
                def nom_traj(ep):
                    p=C.CSTRPlant(seed=seed+100+ep); t=C.CSTRObserverDT(0.05,seed=seed+1)
                    return C.run_cstr_episode(p,t,p.reference(120),sensor_mask=np.ones(3))['residuals']
                def atk_traj_list(a):
                    res=[]
                    for ep in range(10):
                        p=C.CSTRPlant(seed=seed+200+ep); t=C.CSTRObserverDT(0.05,seed=seed+1)
                        ref=p.reference(120)
                        aq=C.CSTR_ATTACK_GENERATORS[a](120,np.random.RandomState(seed+300+ep),E_max=500.0)
                        res.append(C.run_cstr_episode(p,t,ref,aq,np.ones(3))['residuals'])
                    return res
                Hh=120; W_L=8
            else:
                def nom_traj(ep):
                    p=T.CoupledTankPlant(seed=seed+100+ep); t=T.TankObserverDT(0.05,seed=seed+1)
                    return T.run_tank_episode(p,t,p.reference(120),sensor_mask=np.ones(2))['residuals']
                def atk_traj_list(a):
                    res=[]
                    for ep in range(10):
                        p=T.CoupledTankPlant(seed=seed+200+ep); t=T.TankObserverDT(0.05,seed=seed+1)
                        ref=p.reference(120)
                        aq=T.TANK_ATTACK_GENERATORS[a](120,np.random.RandomState(seed+300+ep),E_max=20.0)
                        res.append(T.run_tank_episode(p,t,ref,aq,np.ones(2))['residuals'])
                    return res
                Hh=120; W_L=8
            # forecaster: predict r_k from lags on nominal traj (lag5)
            L=5
            def design(R):
                Z=np.hstack([R[L-l-1:len(R)-l-1] for l in range(L)]); return Z,R[L:]
            Ztr,Ytr=[],[]
            for ep in range(20):
                R=np.nan_to_num(nom_traj(ep)); Z,Y=design(R); Ztr.append(Z); Ytr.append(Y)
            model=Ridge(alpha=1.0).fit(np.vstack(Ztr),np.vstack(Ytr))
            def win_fc_score(R):
                R=np.nan_to_num(R); Z,Y=design(R); e=Y-model.predict(Z)
                k0=Hh//2-W_L//2-L
                seg=e[max(0,k0):max(0,k0)+W_L]
                return np.mean(seg**2)
            # calibration + test negatives from fresh nominal
            cal=[win_fc_score(nom_traj(9000+ep)) for ep in range(30)]  # reuse offset space
            # use distinct offsets for cal vs test
            cal=[win_fc_score(nom_traj(3000+ep)) for ep in range(30)]
            tneg=[win_fc_score(nom_traj(4000+ep)) for ep in range(30)]
            tau=np.percentile(cal,95)
            for a in fams:
                pos=[win_fc_score(R) for R in atk_traj_list(a)]
                y=np.concatenate([np.zeros(len(tneg)),np.ones(len(pos))])
                sc=np.array(tneg+pos)
                acc[a].append(f1_score(y,(sc>tau).astype(int),zero_division=0))
        for a in fams:
            out[f"{bname}_{a}"]=(float(np.mean(acc[a])),float(np.std(acc[a])))
            print(f"  {bname:5s} {LAB[a]:11s} FC-F1={np.mean(acc[a]):.2f}±{np.std(acc[a]):.2f}")
    return out


def bridge_sweep(MP):
    print("\n=== (3) Bridge sweep: correction term + Delta_S + AUC (5 seeds) ===")
    out={}
    for bname,fam in (('CNC','command'),('CSTR','cstr_thermal')):
        for c in (1.0,0.5,0.25,0.1):
            mmds,corrs,dss,aucs=[],[],[],[]
            for seed in cfg.SEEDS:
                mp=MP[bname][seed]; d=mp['data']
                F=(fresh_noms(bname,seed,d,60)-mp['mu'])@mp['V']
                Fcal,Ftest=F[:30],F[30:]
                Xa=(np.nan_to_num(d['atk'][fam]['windows'])-mp['mu'])@mp['V']
                Sn=batch_statevectors(mp['qc'],mp['Xn_p']*c,mp['theta'])
                K=gram_from_sv(Sn); np.fill_diagonal(K,1.0)
                Sa=batch_statevectors(mp['qc'],Xa*c,mp['theta'])
                Kaa=gram_from_sv(Sa); np.fill_diagonal(Kaa,1.0)
                Kna=gram_from_sv(Sn,Sa)
                m=len(K); n=len(Kaa)
                E00=(np.sum(K)-np.trace(K))/(m*(m-1))
                Eaa=(np.sum(Kaa)-np.trace(Kaa))/(n*(n-1))
                mmd=mmd_u(K,Kaa,Kna)
                corr=E00-Eaa
                ds=mmd+corr  # normalized kernel k(z,z)=1
                # held-out AUC
                def sc(X):
                    S=batch_statevectors(mp['qc'],X*c,mp['theta'])
                    return kscore(K,gram_from_sv(Sn,S))
                y=np.concatenate([np.zeros(len(Ftest)),np.ones(len(Xa))])
                auc=roc_auc_score(y,np.concatenate([sc(Ftest),sc(Xa)]))
                mmds.append(mmd);corrs.append(corr);dss.append(ds);aucs.append(auc)
            key=f"{bname}_{c}"
            out[key]=dict(mmd=float(np.mean(mmds)),corr=float(np.mean(corrs)),
                          ds=float(np.mean(dss)),auc=float(np.mean(aucs)))
            print(f"  {bname:5s} c={c:<4} MMD={np.mean(mmds):+.4f}  corr={np.mean(corrs):+.4f}  "
                  f"Delta_S={np.mean(dss):+.4f}  AUC={np.mean(aucs):.2f}")
    return out


def main():
    MP=load_or_build_mp(set()); ensure_tank_stats(MP)
    res={}
    res['surrogate_ext']=ext_surrogate_metrics(MP)
    res['fc']=fc_baseline(MP)
    res['bridge_sweep']=bridge_sweep(MP)
    path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results','v2_fill_remaining.json')
    with open(path,'w') as f: json.dump(res,f,indent=2,default=str)
    print("\nsaved",path)

if __name__=='__main__':
    main()
