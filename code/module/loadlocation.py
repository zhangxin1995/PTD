#!/usr/bin/env python
# coding=utf-8
import json
import numpy as np
# from constant import cam2idx,idx2cam
from math import radians, cos, sin, asin, sqrt
#!/usr/bin/env python
# coding=utf-8

#cam2idx={
#    'SQ0931':0,
#    'SQ0927':1,
#    'SQ0928':2,
#    'SQ0924':3,
#    'SQ0929':4,
#    'SQ0930':5,
#    'SQ0932':6,
#    'SQ0923':7,
#    'SQ0922':8,
#    'SQ0921':9,
#    'SQ0925':10,
#    'SQ0926':11
#    
#}
cam2idx={'SQ0923':0, 
         'SQ0930':1,
         'SQ0927':2,
         'SQ0931':3,
         'SQ0922':4,
         'SQ0925':5,
         'SQ0921':6, 
         'SQ0926':7, 
         'SQ0924':8
}
idx2cam={0:'SQ0923', 
         1:'SQ0930',
         2:'SQ0927',
         3:'SQ0931',
         4:'SQ0922',
         5:'SQ0925',
         6:'SQ0921', 
         7:'SQ0926', 
         8:'SQ0924'
}



def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    dis=2*asin(sqrt(a))*6371*1000
    return dis 
def calcgeodistance(polylines):
    dist=0
    for i in range(1,len(polylines)):
        #dist+=geodistance(polylines[i-1][0],polylines[i-1][1],polylines[i][0],polylines[i][1])
        dist+=geodistance(polylines[i-1][0],polylines[i-1][1],polylines[i][0],polylines[i][1])
    return dist


def loadlocation(filename='location.json'):
    with open(filename) as infile:
         res=json.load(infile)
    mat_durs=np.zeros((9,9))
    mat_dist=np.zeros((9,9))
    index2name=[item['cam_id'] for item in res ]
    name2index={}
    for i,name in enumerate(index2name):
         name2index[name]=i
    for i in range(9):
        for j in range(9):
            mat_durs[i][j]=res[name2index[idx2cam[i]]]['durs'][name2index[idx2cam[j]]]
            mat_dist[i][j]=calcgeodistance(res[name2index[idx2cam[i]]]['polylines'][name2index[idx2cam[j]]])
#    np.savetxt('lcoation_durs.txt',mat_durs)
#    np.savetxt('lcoation_dist.txt',mat_dist)
    mat_total=np.zeros((81,3))
    mat_total[:,0]=range(81)
    mat_total[:,1]=mat_dist.reshape((81))
    mat_total[:,2]=mat_durs.reshape((81))
    return name2index,index2name,mat_total

#loadlocation()


