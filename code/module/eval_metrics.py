from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import copy
import torch
from collections import defaultdict
import sys

try:
    from reid.eval_lib.cython_eval import eval_market1501_wrap
    CYTHON_EVAL_AVAI = True
    print("Cython evaluation is AVAILABLE")
except ImportError:
    CYTHON_EVAL_AVAI = False
    print("Warning: Cython evaluation is UNAVAILABLE")

def gaussian_sp(delta_t, miu, sigma=65):                                                                               
    x, u, sig = delta_t, miu, sigma                                                                                    
    p = np.exp(-(x-u)**2 / (2*sig**2))                                                                                 
    p = max(0.0001, p)                                                                                                 
    return p    

def compute_sp(delta_t, T, sigma=0.7, use_flat=False):   
    sigma_final = max(sigma * T, 5.0)                                                                      
    prob = gaussian_sp(delta_t,T, sigma=sigma_final)                                                                                                                               
    return prob                                                                                                       

def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, N=100):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed N times (default: N=100).
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc, AP = 0., 0.
        for repeat_idx in range(N):
            mask = np.zeros(len(orig_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_orig_cmc = orig_cmc[mask]
            _cmc = masked_orig_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)
            # compute AP
            num_rel = masked_orig_cmc.sum()
            tmp_cmc = masked_orig_cmc.cumsum()
            tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * masked_orig_cmc
            AP += tmp_cmc.sum() / num_rel
        cmc /= N
        AP /= N
        all_cmc.append(cmc)
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_market1501_args(args,distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if q_camids is not None:
        q_camids=np.array(q_camids)
        g_camids=np.array(g_camids)
    q_pids=np.array(q_pids)
    g_pids=np.array(g_pids)
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = args.astype(np.int32)
    print(indices.shape,g_pids.shape)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        if q_camids is not None:
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
            orig_cmc = matches[q_idx][keep] 
        else:
            orig_cmc = matches[q_idx]             
        # compute cmc curve
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
def eval_market1501_Huang(distmat, q_pids, g_pids, q_camids, qts, g_camids, gts, time_mat,max_rank=50, lambda_=50,k=10):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    for i in range(num_q):
        score_st = np.zeros(num_g)
        score=distmat[i,:]
        for j in range(num_g):
            diff=abs(int(qts[i]-gts[j])/1000)
            if diff<time_mat[q_camids[i]][g_camids[j]]:
                score_st[j]=0
            else:
                score_st[j]=k/lambda_*(diff-time_mat[q_camids[i]][g_camids[j]])**(k-1)*np.exp(-((diff-time_mat[q_camids[i]][g_camids[j]])/lambda_)**k)
        distmat[i,:]  = score*score_st
    if q_camids is not None:
        q_camids=np.array(q_camids)
        g_camids=np.array(g_camids)
    q_pids=np.array(q_pids)
    g_pids=np.array(g_pids)
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1).astype(np.int32)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        if q_camids is not None:
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
            orig_cmc = matches[q_idx][keep] 
        else:
            orig_cmc = matches[q_idx]             
        # compute cmc curve
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP


def eval_market1501_Xie(distmat, q_pids, g_pids, q_camids, qts, g_camids, gts, time_mat,max_rank=50):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    for i in range(num_q):
        score_st = np.zeros(num_g)
        score=distmat[i,:]
        for j in range(num_g):
            diff=abs(int(qts[i]-gts[j])/1000)
            m=time_mat[q_camids[i]][g_camids[j]]
            score_st[j]=1.0/compute_sp(diff,m)
        distmat[i,:] = score*score_st
    if q_camids is not None:
        q_camids=np.array(q_camids)
        g_camids=np.array(g_camids)
    q_pids=np.array(q_pids)
    g_pids=np.array(g_pids)
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1).astype(np.int32)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        if q_camids is not None:
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
            orig_cmc = matches[q_idx][keep] 
        else:
            orig_cmc = matches[q_idx]             
        # compute cmc curve
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP
def eval_market1501_zhang(distmat, q_pids, g_pids, q_camids, qts, g_camids, gts, func,qcs,gcs,max_rank=50, lambda1=1,lambda2=1,alpha1=20,alpha2=1e-2,thresh=0.9,topk=10,hist=None):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1).astype(np.int32)
    for i in range(num_q):
        score_st = np.zeros(topk)
        args=indices[i,:topk]
        score=distmat[i,args]
        #for j in range(num_g):
        for k,j in enumerate(args):
            score_st[k]=func(qcs[i],qts[i],gcs[j],gts[j])
            if score_st[k]<thresh:
                score_st[k]=0
        score_st= 1/(1+lambda1*np.exp(-alpha1*score))*1/(1+lambda2*np.exp(alpha2*score_st))
        #distmat[i,:]  = 1/(1+lambda1*np.exp(-alpha1*score))*1/(1+lambda2*np.exp(-alpha2*score_st))
        indices[i,:topk]=args[np.argsort(score_st)]
    if q_camids is not None:
        q_camids=np.array(q_camids)
        g_camids=np.array(g_camids)
    q_pids=np.array(q_pids)
    g_pids=np.array(g_pids)
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        if q_camids is not None:
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
            orig_cmc = matches[q_idx][keep] 
        else:
            orig_cmc = matches[q_idx]             
        # compute cmc curve
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP





def eval_market1501_wang(distmat, q_pids, g_pids, q_camids, qts, g_camids, gts, hist,max_rank=50, interval=10,lambda1=1,lambda2=2,alpha1=5,alpha2=5,para=0):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    for i in range(num_q):
        score_st = np.zeros(num_g)
        score=distmat[i,:]
        for j in range(num_g):
            noise=np.random.randint(para*60*10)*1000
            if qts[i]>gts[j]:
                diff=int(int(qts[i]-gts[j]+noise)/1000/interval)
                if diff>=hist[g_camids[j]][q_camids[i]].shape[0]:
                    score_st[j]=0
                else:
                    score_st[j]=hist[g_camids[j]][q_camids[i]][diff]
            else:
                diff=int(int(-qts[i]+gts[j]+noise)/1000/interval)
                if diff>=hist[q_camids[i]][g_camids[j]].shape[0]:
                    score_st[j]=0
                else:
                    score_st[j]=hist[q_camids[i]][g_camids[j]][diff]
        distmat[i,:]  = 1/(1+lambda1*np.exp(-alpha1*score))*1/(1+lambda2*np.exp(alpha2*score_st))
    if q_camids is not None:
        q_camids=np.array(q_camids)
        g_camids=np.array(g_camids)
    q_pids=np.array(q_pids)
    g_pids=np.array(g_pids)
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1).astype(np.int32)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        if q_camids is not None:
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
            orig_cmc = matches[q_idx][keep] 
        else:
            orig_cmc = matches[q_idx]             
        # compute cmc curve
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False, use_cython=True):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    else:
        if use_cython and CYTHON_EVAL_AVAI:
            return eval_market1501_wrap(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
        else:
            return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)

def evaluate_args(args, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_cython=True):
    return eval_market1501_args(args,distmat, q_pids, g_pids, q_camids, g_camids, max_rank)

def evaluate_group_search(distmat,qls,threshold):
    tpr=0
    tre=0
    qls=np.array(qls)
    t=0
    t1=0
    t2=0
    t3=0
    t4=0
    t5=[]
    t6=0
    gs=set(qls)
    for g in qls:
        idxs=np.where(qls==g)[0]
        for i in idxs:
            for j in idxs:
                if i!=j:
                    print(g,i,j,distmat[i,j])
    print('qls size:',len(set(qls)))
    num_r=0
    for i in range(distmat.shape[0]):
        idxs=np.where(distmat[i,:]<threshold)[0]
       # print(distmat[i,:])
        q=qls[i]
        g=np.where(qls==q)[0]
        #print('g',g,'idxs',idxs)
        #print('q',q,'qls[idxs]',qls[idxs],'qls[g]',qls[g])
        g=set(g)
        idxs=set(idxs)
        idxs.remove(i)
        g.remove(i)
        if len(g)>0:
            t5.append(q)
        if len(idxs)==0 and len(g)==0:
            t1+=1
            pr=1
            re=-1
        elif len(idxs)==0 and len(g)!=0:
            t6+=1
            t2+=1
            pr=0
            re=0
        elif len(idxs)!=0 and len(g)==0:
            t3+=1
            pr=0
            re=-1
        else:
            t6+=1
            pr=len(idxs.intersection(g))/len(idxs)
            re=len(idxs.intersection(g))/len(g)
            t4+=1
        tpr+=pr
        if re!=-1:
            tre+=re
            num_r+=1
    t=t4+t2+t3+t1
    #print('t',t1,t2,t3,t4)
    #print(len(set(t5)),t6)
    print(tpr,t)
    tpr/=t #distmat.shape[0]
    tre/=num_r #distmat.shape[0]
    f1=2*tpr*tre/(tpr+tre)
    return tpr,tre,f1,t



def eval_trajectory_retrieval(distmat, q_pids, g_pids, max_rank=50):
    num_q, num_g = distmat.shape
    q_pids=np.array(q_pids)
    g_pids=np.array(g_pids)
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1).astype(np.int32)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        orig_cmc = matches[q_idx]             
        # compute cmc curve
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP


 
def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if q_camids is not None:
        q_camids=np.array(q_camids)
        g_camids=np.array(g_camids)
    q_pids=np.array(q_pids)
    g_pids=np.array(g_pids)
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1).astype(np.int32)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        if q_camids is not None:
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
            orig_cmc = matches[q_idx][keep] 
        else:
            orig_cmc = matches[q_idx]             
        # compute cmc curve
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False, use_cython=True):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    else:
        if use_cython and CYTHON_EVAL_AVAI:
            return eval_market1501_wrap(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
        else:
            return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)

def evaluate_args(args, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_cython=True):
    return eval_market1501_args(args,distmat, q_pids, g_pids, q_camids, g_camids, max_rank)


def average_search_time(distmat,qls,idx2pathidx,tpath2index,cluster2index,topk=20,indices=None):
    loss=0
    for i in range(len(qls)):
        if indices is None:
            ag=np.argsort(distmat[i])[:topk]
        else:
            ag=indices[i][:topk]
        top=np.min([topk,distmat[i].shape[0]])
        k=0
        for j in range(top):
            for key in idx2pathidx[qls[i]]:
                paths=tpath2index[key]
                tloss=set(paths)&set(cluster2index[ag[j]])
                if len(tloss)>0:
                    k=j
        loss+=k
    loss/=len(qls)
    return loss



def rankscore(distmat,qls,idx2pathidx,tpath2index,cluster2index,topk=20,indices=None):
    loss=0
    loss2=0
    total=0
    loss3=[]
    #print(distmat.shape,len(cluster2index))
    for i in range(len(qls)):
        if indices is None:
            ag=np.argsort(distmat[i])[:topk]
        else:
            ag=indices[i][:topk]
        tloss=0
        weight=defaultdict(int)
        top=np.min([topk,distmat[i].shape[0]])
        for j in range(top):
            for k in cluster2index[ag[j]]:
                weight[k]+=1
                #if k not in weight.keys():
                #    weight[k]=1
                #else:
                #    weight[k]+=1
        for key in weight.keys():
            loss2+=weight[key]
        total+=len(weight.keys())
        temp=0
        for j in range(top):
            tloss=0
            for key in idx2pathidx[qls[i]]:
                paths=tpath2index[key]
                tloss+=jsp(paths,cluster2index[ag[j]],weight)
            temp+=tloss/((j+1)*(j+1)) #   loss+=1.0/((j+1)*(j+1))*jsp(paths,cluster2index[ag[j]])
        loss3.append(temp)
        loss+=temp
        if i==0 and False:
            print('====================================================')
            print(loss,temp)
            for key in idx2pathidx[qls[i]]:
                paths=tpath2index[key]
                print(paths)
            print('****************************************************')
            for j in range(top):
                print(cluster2index[ag[j]],jsp(paths,cluster2index[ag[j]],weight))
            input()
    loss/=len(qls)
    loss2/=total
    return loss,loss2

def tds(distmat,qls,idx2pathidx,tpath2index,cluster2index,topk=20,threshold=0.04,indices=None):
    tloss=0
    for i in range(len(qls)):
        if indices is None:
            #ag=np.argsort(distmat[i])[:topk]
            ag=np.where(distmat[i,:]<threshold)[0]
        else:
            ag=np.where(distmat[i,:]<threshold)[0]
            #ag=indices[i][:topk]
        #top=np.min([topk,distmat[i].shape[0]])
        num=len(idx2pathidx[qls[i]])
        pr=0
        for j in range(ag.shape[0]):
            flag=False
            for key in idx2pathidx[qls[i]]:
                paths=tpath2index[key]
                if path_is_matched(paths,cluster2index[ag[j]]):
                    flag=True
                    break
            if flag:
                pr+=1
                continue
        tloss+=pr*1.0/num
    tloss/=len(qls)
    return tloss


def path_is_matched(a,b):
    if len(a)!=len(b):
        return False
    inds=set(a)&set(b)
    if len(inds)==len(a):
        return True

def jsp(a,b,weight):
    inds=set(a)&set(b)
    loss=0
    for item in inds:
        loss+=1.0/weight[item]
    loss/=len(set(a)|set(b))
    return loss

