#!/usr/bin/env python
# coding=utf-8
import numpy as np
import time
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt 
from .constant import cam2idx,idx2cam 
from sklearn.ensemble.partial_dependence import plot_partial_dependence,partial_dependence
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import naive_bayes as nb
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from collections import defaultdict
import torch
from torch import nn
from torch import optim
torch.set_printoptions(profile="full")
class SpatialModel(object):
    def __init__(self,**args):
        pass
    def predict(self,):
        pass
    def predict_proba(self,):
        pass
    def fit(self,x,y):
        pass

class GaussianMLP(nn.Module):
    def __init__(self,middle=100):
        self.act=nn.Sigmod()
        self.input=nn.Linear(2,middle)
        self.hidden=nn.Linear(middle,middle)
        self.output=nn.Linear(middle,2)
        self.model=nn.Sequential([self.input,self.act,self.hidden,self.act,self.output])

    def forward(self,xs,ds,labels):
        pass

class GaussianPoly(nn.Module):
    def __init__(self,n1,n2):
        super(GaussianPoly,self).__init__()
        self.alpha=nn.Parameter(torch.randn((n1),requires_grad=True))
        self.beta=nn.Parameter(torch.randn((n2),requires_grad=True))
        self.ce=nn.CrossEntropyLoss()
        self.act=nn.Sigmoid()
    def predict_us(self,d):
        temp=np.zeros(n1)
        for i in range(1,n1):
            temp[i]=temp[i-1]*d
        u=torch.sum(torch.Tensor(temp)*self.alpha)
        s=torch.sum(torch.Tensor(temp)*self.beta)
        return us

    def predict_prob(self,x,d):
        u,s=self.predict_us(d)
        return torch.exp((x-u)**2/s**2)/(torch.sqrt(2*np.pi)*s)

    def test(self,x,y):
        pass

        

    def forward(self,xs,ds,labels):
        ds=np.array(ds)
        xs=np.array(xs)
        labels=torch.Tensor(labels)
        unique_d=np.unique(ds)
        idx=np.zeros(ds.shape)
        n1=self.alpha.size()[0]
        n2=self.beta.size()[0]
        prob=torch.zeros((xs.shape[0],2))
        params=[]
        param_us=torch.zeros(xs.shape)
        param_ss=torch.zeros(xs.shape)
        loss=0
        print(unique_d)
        for i,d in enumerate(unique_d):
            args=np.where(ds==d)
            idx[args]=i
            temp=np.zeros(n1)
            temp[0]=1
            for i in range(1,n1):
                temp[i]=temp[i-1]*d
            u=torch.sum(torch.Tensor(temp[:n1])*torch.exp(self.alpha))
            s=torch.sum(torch.Tensor(temp[:n2])*torch.exp(self.beta))
#            print(temp,u,s)
#            print(u,s,args)
#            print(temp.shape,self.alpha.size(),u.size(),s.size())
            param_us[args]=u
            param_ss[args]=s
        xs=torch.Tensor(xs)
        ds=torch.Tensor(ds)
        param_us=torch.Tensor(param_us)
        param_ss=torch.Tensor(param_ss)
        dt=1
#        print(self.alpha,self.beta)
        prob[:,1]=self.act(torch.exp(-(xs-param_us)**2/param_ss**2)/((torch.sqrt(torch.tensor(2*3.141592657))*param_ss)+1e-8))
        prob[:,0]=1-prob[:,1]
#        prob=prob.reshape((prob.shape[0],1))
        labels=labels.long()
#        print(prob.shape,labels.shape)
        S=self.ce(prob,labels)
        auc=metrics.roc_auc_score(labels.detach().numpy().astype(np.int),prob.detach().numpy()[:,1])
        print('auc',auc)
        #S=torch.sum((xs-param_us)**2/(param_ss**2)*labels)+torch.sum(torch.log(param_ss)*labels)+dt*(torch.sum(self.alpha**2)+1e-1*torch.sum(self.beta**2))
        #S/=labels.sum() #xs.size()[0]
        print('Loss',S)
        return S

class GaussianSpatialModel(SpatialModel):
    def __init__(self,**args):
        super(GaussianSpatialModel,self).__init__(**args)
        self.distance2y=defaultdict(list)
        self.distance2hist={}
    def fit(self,x,y):
        for i,k in enumerate(x):
            self.distance2y[k[0]].append(k[1])
        fig=plt.figure()
        ax=fig.gca(projection='3d')
        for key in self.distance2y.keys():
            self.distance2hist[key],edges=np.histogram(self.distance2y[key],bins=20)
            q=np.array([key for edge in edges[:-1]])
            #print(q.shape,edges.shape)
            ax.plot(q,edges[:-1],self.distance2hist[key]*1.0/np.sum(self.distance2hist[key]))
        ax.view_init(elev=35., azim=35)
        ax.set_ylabel('time difference')
        ax.set_xlabel('distance')
        ax.set_ylim(0,30000)
        ax.set_xlim(0,800)
        ax.set_zlim(0,1)
        ax.set_zlabel('frequency')
        plt.savefig('1234.svg',dpi=3000, bbox_inches = 'tight')
        #print('123 finished')
        #exit()
#        print(self.distance2hist)

class GaussianModel(SpatialModel):
    def __init__(self,**args):
        if 'n1' in args.keys():
            self.n1=args['n1']
        else:
            self.n1=4
        if 'n2' in args.keys():
            self.n2=args['n2']
        else:
            self.n2=4
        self.model=GaussianPoly(self.n1,self.n2)
        self.iters=10000
        self.optimizer = optim.SGD(self.model.parameters(), lr = 1e-1, momentum = 0.9)

    def fit(self,x,y):
        ss=[t[0] for t in x]
        xs=[t[1] for t in x]
        ls=[]
        for l in y:
            if l==1:
                ls.append(l)
            else:
                ls.append(0)
        y=ls
        for i in range(self.iters):
            self.optimizer.zero_grad()
            loss=self.model.forward(xs,ss,y)
            loss.backward()
            self.optimizer.step()
        exit(0)
    def predict_proba(self,x):
        pass

class Model:
    def __init__(self,model_name,**args):
        names2model_fun={'decision_tree':DecisionTreeClassifier,
                         'adaboost':AdaBoostClassifier,
                         'gbrt':GradientBoostingClassifier,
                         'svm':SVC,
                         'gaussianNB':nb.GaussianNB,
                         'multinomialNB':nb.MultinomialNB,
                         'bernoulliNB':nb.BernoulliNB,
                         'MLP':MLPClassifier,
                         'gauss_poly':GaussianModel}
        if model_name not in names2model_fun:
            print('model_name should be in name2model_fun!')
            exit(0)
        self.model=names2model_fun[model_name](**args)
        self.model_name=model_name


    def predict(self,x):
        return self.model.predict_proba(x)

    def fit(self,x,y):
        self.model.fit(x,y)

    def score(self,x,y):
        return self.model.score(x,y)

    def distance_predict(self,dist,x):
        dist=np.ones(x.shape[0])*dist.reshape((x.shape[0],1))
        feature=np.hstack(x,dist)
        return self.model.predict(feature)

    def distance_score(self,dist,x,y):
        dist=np.ones(x.shape[0])*dist.reshape((x.shape[0],1))
        feature=np.hstack(x,dist)
        return self.model.score(feature,y)

    def distance_fit(self,dist,x,y):
        dist=np.ones(x.shape[0])*dist.reshape((x.shape[0],1))
        feature=np.hstack(x,dist)
        self.model.fit(feature,y)


class Experiment:
    def __init__(self,train_x,train_y,test_x,test_y,model_name,mode,locationmat,args,**kwargs):
        self.model_name=model_name
        self.train_x=train_x
        self.train_y=train_y
        self.test_x=test_x
        self.test_y=test_y
        self.locationmat=locationmat
        self.args=args
        self.save_dir=args.save_dir
        self.mode=mode
        self.kwargs=kwargs

    def save_model(self,mode):
        with open('{}/{}_{}.pkl'.format(self.save_dir,mode,self.model_name),'wb') as out:
            print('{}/{}_{}.pkl'.format(self.save_dir,mode,self.model_name))
            pkl.dump(self.model,out)

    def save_csv(self,name,text):
        path='{}/{}.csv'.format(self.save_dir,name)
        np.savetxt(path,text)

    def visual_data(self,index):
        plt.figure()
        i=int(index//9)
        j=int(index%9)
#        print(i,j)
#        print(idx2cam[i],idx2cam[j])
        colors=[0,1]
        
        num=1
        d=self.train_x[i][j]
        y=self.train_y[i][j]
        c=[colors[item] for item in y]
        x=[num*k for k in c]
        plt.scatter(d,x,c=c)

        num=2
        colors=[3,5]
        d=self.test_x[i][j]
        y=self.test_y[i][j]
        c=[colors[item] for item in y]
        x=[num*k+2 for k in c]
        plt.scatter(d,x,c=c)
        plt.show()

    def random_train(self,seed,num):
        start = time.perf_counter()
        self.model=Model(self.model_name,**self.kwargs)
        np.random.seed(seed)
        pairs=[]
        for i in range(9):
            for j in range(i+1,9):
                pairs.append((i,j))
        #indexs=range(len(pairs))# random.sample(range(len(pairs)),num)
        indexs=random.sample(range(len(pairs)),num)
        print(indexs)
        #train_idx=[pairs[i] for i in indexs]
        train_idx=[[2,5],[5,7],[2,8],[2,4],[0,2],[7,8],[2,7],[5,8],[4,6],[0,6],[0,4],[7,7],[0,0],[6,6],[4,4],[2,2],[5,5],[8,8]]
        r2_score_mat=np.zeros((9,9))
        auc_mat=np.zeros((9,9))
        train_x=[]
        train_y=[]
        test_x=[]
        test_y=[]
        flag=np.zeros((9,9))
        for idx in train_idx:
            flag[idx[0]][idx[1]]=1
            flag[idx[1]][idx[0]]=1
            for i,item in enumerate(self.train_x[idx[0]][idx[1]]):
                train_x.append([self.locationmat[idx[0]][idx[1]],item])
                train_y.append(self.train_y[idx[0]][idx[1]][i])
        for i in range(9):
            for j in range(9):
                for k,item in enumerate(self.test_x[i][j]):
                    test_x.append([self.locationmat[i][j],item[0]])
                    test_y.append(self.test_y[i][j][k])
        sp=GaussianSpatialModel()
        sp.fit(train_x,train_y)
        #sp.fit(test_x,test_y)
        exit()
        self.model.fit(train_x,train_y)
        end = time.perf_counter()
        print("训练运行时间为", round(end-start), 'seconds')
        args=np.where(np.array(test_y)==1)
        feature=np.array(test_x)[args]
        ypred=self.model.predict(train_x)[:,1]
        auc_mat=metrics.roc_auc_score(train_y,ypred)
        print(self.model_name,1,auc_mat)
        ypred=self.model.predict(test_x)[:,1]
        auc_mat=metrics.roc_auc_score(test_y,ypred)
        print(self.model_name,2,auc_mat)
        #self.save_model('group')
        #exit()
        return self.model
        
    def dist_go(self,idxs=None):
        self.model=Model(self.model_name,**self.kwargs)
        self.visual_data(23)
        if idxs==None:
            train_idx=[[2,5],[5,7],[2,8],[2,4],[0,2],[7,8],[2,7],[5,8],[4,6],[0,6],[0,4],[7,7],[0,0],[6,6],[4,4],[2,2],[5,5],[8,8]]
        else:
            train_idx=idxs
        r2_score_mat=np.zeros((9,9))
        auc_mat=np.zeros((9,9))
        train_x=[]
        train_y=[]
        test_x=[]
        test_y=[]
        flag=np.zeros((9,9))
        for idx in train_idx:
            flag[idx[0]][idx[1]]=1
            flag[idx[1]][idx[0]]=1
            for i,item in enumerate(self.train_x[idx[0]][idx[1]]):
                train_x.append([self.locationmat[idx[0]][idx[1]],item])
                train_y.append(self.train_y[idx[0]][idx[1]][i])
        for i in range(9):
            for j in range(9):
                #for k,item in enumerate(self.train_x[i][j]):
                #    test_x.append([self.locationmat[i][j],item])
                #    test_y.append(self.train_y[i][j][k])
                for k,item in enumerate(self.test_x[i][j]):
                    test_x.append([self.locationmat[i][j],item[0]])
                    test_y.append(self.test_y[i][j][k])
        self.model.fit(train_x,train_y)

        #args=np.where(np.array(train_y)==1)
        #feature=np.array(train_x)[args]
        #sp=GaussianSpatialModel()
        #sp.fit(feature,train_y)



        args=np.where(np.array(test_y)==1)
        feature=np.array(test_x)[args]
        #sp=GaussianSpatialModel()
        #sp.fit(feature,test_y)
        #print(test_x[0])
        #r2_score=self.model.score(test_x,test_y)

        plt.figure()
        ypred=self.model.predict(train_x)[:,1]
        auc_mat=metrics.roc_auc_score(train_y,ypred)
        print(self.model_name,1,auc_mat)
        ypred=self.model.predict(test_x)[:,1]
        auc_mat=metrics.roc_auc_score(test_y,ypred)
        print(self.model_name,2,auc_mat)
        fpr, tpr, thersholds = roc_curve(test_y, ypred, pos_label=1)
        plt.plot(fpr, tpr,  label='ROC (area = {0:.2f})'.format(auc_mat), lw=2)
        with open('{}_roc.pkl'.format(self.model_name),'wb') as outp:
            pkl.dump({self.model_name:{'fpr':fpr,'tpr':tpr,'auc':auc_mat}},outp)
        plt.savefig('{}_roc.jpg'.format(self.model_name))
        plt.figure()


        answer=np.zeros((ypred.shape[0],2))
        answer[:,0]=ypred
        answer[:,1]=test_y
#·        np.savetxt('./logs/answer.csv',answer)
        #self.save_csv('answer',answer)
        plt.figure()
        colors=['r','b']
        x=[item[0] for item in train_x]
        y=[item[1] for item in train_x]
        c=[colors[item] for item in train_y]
        plt.scatter(x,y,c=c)
        plt.savefig('a.jpg')
        plt.figure()

        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #k=defaultdict(list)
        #for i,item in enumerate(x):
        #    k[item].append(y[i])
        #ax.plot(x, y, z)



        x=[item[0] for item in test_x]
        y=[item[1] for item in test_x]
        c=[colors[item] for item in test_y]
        plt.scatter(x,y,c=c)
        plt.savefig('b.jpg')

        x=np.arange(0,400,10)
        y=np.arange(0,2000,20)
        X,Y=np.meshgrid(x,y)
        XX=X.flatten()
        YY=Y.flatten()
        fx=np.vstack([XX,YY]).transpose()

        #plot_partial_dependence(self.model.model,test_x,features=[0,1,(0,1)],feature_names=['distance','diff_time'])
        #plt.savefig('{}_partial.jpg'.format(self.model_name))

        fig=plt.figure(figsize=(12,12))
        ax=fig.add_subplot(2,1,1,projection='3d')
        ax.set_top_view()
#        help(partial_dependence)
        #pdp,axes=partial_dependence(self.model.model,X=fx,target_variables=[[0,1]],grid_resolution=2)
 #       pdp,axes=partial_dependence(self.model.model,X=fx,target_variables=[[0,1]])
        #pdp=self.model.model.predict(fx)
        pdp=self.model.model.predict_proba(fx)[:,1]
        pdp=pdp.reshape(X.shape)
        ax.view_init(elev=30., azim=45)
 #       X,Y=np.meshgrid(axes[0],axes[1])
 #       print(X.shape,Y.shape,pdp.shape)
        ax.plot_surface(X,Y,pdp*1000,cmap=plt.cm.Spectral,cstride=1,rstride=1)
 #       ax=fig.add_subplot(2,1,2)
        plt.xlabel('distance')
        plt.ylabel('time difference')
        plt.contour(X,Y,pdp*1000,cmap=plt.cm.Spectral)
        plt.savefig('{}_f.jpg'.format(self.model_name))
        self.save_model('UnionModel')
        #with open('./logs/UnionModel_{}.pkl'.format(self.model_name),'wb') as out:
        #    pkl.dump(self.model,out)
        with open('{}_f.txt'.format(self.model_name),'w') as out:
            for i in range(X.shape[0]):
                for j in range(X[i].shape[0]):
                    out.write('{} {} {}\n'.format(X[i][j],Y[i][j],pdp[i][j]))
                out.write('\n')
        return  self.model

    def go(self):
        self.model={}
        for i in range(9):
            if i not in self.model.keys():
                self.model[i]={}
            for j in range(9):
                self.model[i][j]=Model(self.model_name,**self.args)
        r2_score_mat=np.zeros((9,9))
        auc_mat=np.zeros((9,9))
        train_true_num=np.zeros((9,9))
        train_false_num=np.zeros((9,9))
        test_true_num=np.zeros((9,9))
        test_false_num=np.zeros((9,9))
        self.visual_data(23)
        mean_mat=np.zeros((9,9))
        median_mat=np.zeros((9,9))
        max_mat=np.zeros((9,9))
        min_mat=np.zeros((9,9))
        hist_max_mat=np.zeros((9,9))
        bins=np.arange(0,2000,50)

        for i in range(9):
            for j in range(9):
                tidx=np.where(np.array(self.train_y[i][j])==1)
                #print(self.train_y[i][j],tidx)
                if tidx[0].shape[0]>0:
                    max_mat[i][j]=np.max(np.array(self.train_x[i][j])[tidx])
                    min_mat[i][j]=np.min(np.array(self.train_x[i][j])[tidx])
                    hist=plt.hist(np.array(self.train_x[i][j])[tidx],bins=bins)
                    hist_max_mat[i][j]=hist[1][np.argmax(hist[0])]+25
                else:
                    hist_max_mat[i][j]=-1
                    max_mat[i][j]=-1
                    min_mat[i][j]=-1
                mean_mat[i][j]=np.mean(np.array(self.train_x[i][j])[tidx])
                median_mat[i][j]=np.median(np.array(self.train_x[i][j])[tidx])
                train_true_num[i][j]+=sum(self.train_y[i][j])
                train_false_num[i][j]+=sum(1-np.array(self.train_y[i][j]))
                test_false_num[i][j]+=sum(1-np.array(self.test_y[i][j]))
                test_true_num[i][j]+=sum(self.test_y[i][j])
                if len(self.train_x[i][j])==0 or len(self.test_y[i][j])==0:
                    r2_score_mat[i][j]=-1
                    auc_mat[i][j]=-1
                    continue
                self.model[i][j].fit(np.array(self.train_x[i][j]).reshape((len(self.train_x[i][j]),1)),self.train_y[i][j])
                r2_score_mat[i][j]=self.model[i][j].score(self.test_x[i][j],self.test_y[i][j])
                ypred=self.model[i][j].predict(self.test_x[i][j])
                try:    
                    auc_mat[i][j]=metrics.roc_auc_score(self.test_y[i][j],ypred)
                except:
                    auc_mat[i][j]=-1
        #np.savetxt('./logs/train_hist_max.txt',hist_max_mat.reshape(81),fmt='%.03f')
        #np.savetxt('./logs/train_max.txt',max_mat.reshape(81),fmt='%.03f')
        #np.savetxt('./logs/train_min.txt',min_mat.reshape(81),fmt='%.03f')
        #np.savetxt('./logs/train_mean.txt',mean_mat.reshape(81),fmt='%.03f')
        #np.savetxt('./logs/train_median.txt',median_mat.reshape(81),fmt='%.03f')
        #np.savetxt('./logs/r2_score_mat.csv',r2_score_mat,fmt='%.03f')
        #np.savetxt('./logs/auc_score_mat.csv',auc_mat,fmt='%.03f')
        total=np.zeros((81,6))
        total[:,0]=train_true_num.reshape((81))
        total[:,1]=train_false_num.reshape((81))
        total[:,2]=test_true_num.reshape((81))
        total[:,3]=test_false_num.reshape((81))
        total[:,4]=auc_mat.reshape((81))
        total[:,5]=r2_score_mat.reshape(81)

        self.save_csv('total',total)
       # np.savetxt('./logs/total.csv',total)
        #self.save_model('Seperate')
        return model

       # with open('./logs/Seperate_{}.pkl'.format(self.model_name),'wb') as out:
       #     pkl.dump(self.model,out)
