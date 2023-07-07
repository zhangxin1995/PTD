import os.path as osp
import sys
import pickle as pkl
import numpy as np
class CompareTimeByCamera(object):
    def __init__(self,camera_timemat,camera_name2idx,camera_distmat,convertname=True,**args):
        self.cd=camera_distmat
        self.ct=camera_timemat
        self.name2idx=camera_name2idx
        self.cam2id=camera_name2idx
        self.convertname=convertname
    def __call__(self,ta,tcsa,tb,tcsb):
        pass
class CompareTimeByCameraFromThreshhold(CompareTimeByCamera):
    def __init__(self,camera_timemat,camera_name2idx,camera_distmat,**args):
        super(CompareTimeByCameraFromThreshhold,self).__init__(camera_timemat,camera_name2idx,camera_distmat,**args)
        self.alpha=args['alpha']             
        self.beta=args['beta'] 
        self.cd=camera_distmat.reshape((9,9)) 
        self.ct=camera_timemat.reshape((9,9))
        self.name2idx=camera_name2idx         
    def __call__(self,tcsa,ta,tcsb,tb):
        if self.convertname:
            idxa=self.name2idx[tcsa]
            idxb=self.name2idx[tcsb]
        else:
            idxa=tcsa
            idxb=tcsb
        if not isinstance(ta,list):
            ab=abs(ta-tb)/1000.0
        else:
            ab1=min(ta)-max(tb)
            ab2=max(ta)-min(tb)
            ab=min(ab1,ab2)
        if self.ct[idxa][idxb]*self.alpha<ab and ab<self.beta*self.ct[idxa][idxb]:
            return True
        else:
            return False

class CompareTimeByCameraFromSperateModel(CompareTimeByCamera):
    def __init__(self,camera_timemat,camera_name2idx,camera_distmat,convertname,**args):
        super(CompareTimeByCameraFromSperateModel,self).__init__(camera_timemat,camera_name2idx,camera_distmat,convertname,**args)
        self.alpha=args['alpha']
        self.beta=args['beta']
        self.model=args['model']
    def compare(self,ta,tcsa,tb,tcsb):
        if self.convertname:
            idxa=self.name2idx[tcsa]
            idxb=self.name2idx[tcsb]
        else:
            idxa=tcsa
            idxb=tcsb
        if type(ta)!='list':
            ab=abs(ta-tb)/1000.0
        else:
            ab1=min(ta)-max(tb)
            ab2=max(ta)-min(tb)
            ab=min(ab1,ab2)
        if self.model[idxa][idxb](ab):
            return True
        else:
            return False

class CompareTimeByCameraFromUnionModel(CompareTimeByCamera):
    def __init__(self,camera_timemat,camera_name2idx,camera_distmat,convertname,**args):
        super(CompareTimeByCameraFromUnionModel,self).__init__(camera_timemat,camera_name2idx,camera_distmat,convertname,**args)
        self.model=args['model']
        self.cd=camera_distmat.reshape((9,9))
        if 'thresh' in args.keys():
            self.thresh=args['thresh']
        else:
            self.thresh=0.9
    def set_model(self,model):
        self.model=model
    def __call__(self,tcsa,ta,tcsb,tb,prob=False):
        if self.convertname:
            idxa=self.name2idx[tcsa]
            idxb=self.name2idx[tcsb]
        else:
            idxa=tcsa
            idxb=tcsb

        if type(ta)!='list':
            ab=abs(ta-tb)/1000.0
        else:
            ab1=min(ta)-max(tb)
            ab2=max(ta)-min(tb)
            ab=min(ab1,ab2)
        x=np.array((self.cd[idxa][idxb],ab)).reshape((1,-1))
        #print(x)
        if self.thresh is None:
            y=self.model.predict(x)[:,1]
            #y=self.model.model.predict_proba(x)[:,1]
        else:
            y_score=self.model.model.predict_proba(x)[:,1]
        #    print(y_score,self.thresh)
            if self.thresh==0 or prob:
                return y_score
            elif y_score>self.thresh:
                y=1
            else:
                y=-1
        return y

class CompareTimeByCameraHistgram(CompareTimeByCamera):
    def __init__(self,camera_timemat,camera_name2idx,camera_distmat,convertname,**args):
        super(CompareTimeByCameraHistgram,self).__init__(camera_timemat,camera_name2idx,camera_distmat,convertname,**args)
        self.model=args['model']
        self.cd=camera_distmat.reshape((9,9))
        if 'thresh' in args.keys():
            self.thresh=args['thresh']
        else:
            self.thresh=0.9
        self.interval=args['interval']
    def __call__(self,tcsa,ta,tcsb,tb,prob=False):
        diff=abs(int(int(ta-tb)/1000.0/self.interval))
        if diff>self.model[self.name2idx[tcsa]][self.name2idx[tcsb]].shape[0]:
            score_st=0
        else:
            score_st=self.model[self.name2idx[tcsa]][self.name2idx[tcsb]][diff]
        return score_st*100
        
compare_factory={
    'UM':CompareTimeByCameraFromUnionModel,
    'TH':CompareTimeByCameraFromThreshhold,
    'SM':CompareTimeByCameraFromSperateModel,
    'HIST':CompareTimeByCameraHistgram
}
compare_parameters={
    'UM':{},
    'TH':{'alpha':0.5,'beta':5},
    'SM':{}
}

def generate_compare(cn,modelname,locationmat,name2idx,pathmodel_dir,convertname=True,**args):
    if cn not in compare_factory.keys():
        raise('Unknow Spatial Model')
    compare_parameters[cn].update(args)
    compare=compare_factory[cn](locationmat[:,2],name2idx,locationmat[:,1],model=loadmodel(pathmodel_dir,'{}.pkl'.format(modelname)),convertname=convertname,**compare_parameters[cn])
    return compare

def loadmodel(root,name):
    p=osp.join(root,name)
    with open(p,'rb') as infile:
        if sys.version[0]!='3':
            res=pkl.load(infile)
        else:
            res=pkl.load(infile,encoding='latin')
    return res

class SpatialTemporalModel(object):
    def __init__(self,path_dataset,pathmodel_dir,mode,modelname,convertname=True):
        self.path_dataset=path_dataset
        self.pathmodel_dir=pathmodel_dir
        self.name2idx,self.idx2name,self.lm=loadlocation(osp.join(self.path_dataset.root,'location.json'))
        self.convertname=convertname
        self.loadmodel(mode,modelname)

    def loadmodel(self,mode,modelname):
        self.mode=mode
        self.modelname=modelname
        self.compare=generate_compare(mode,modelname,self.lm,self.name2idx,self.pathmodel_dir,convertname=self.convertname)

    def __call__(self,c1,t1,c2,t2):
        return self.compare(c1,t1,c2,t2)

    def set_model(self,model):
        self.compare.set_model(model)