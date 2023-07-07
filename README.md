# Person Trajectory Dataset



## Description
Person Trajectory Dataset(PTD) comes from paper `Cross-Camera Trajectory Helps Person Retrieval in a Camera Network`. This is collected from a camera network of nine cameras from 8 a.m. to 8 p.m. The spatial distribution of the cameras for this dataset is shown in Fig. 1. The person image sequence acquired are preliminarily labeled by FPN target detection and DeepSort tracking, and then the annotation results are corrected manually. The images of any person that only appears under one camera are also eliminated to ensure the persons in PTD have appeared under at least two different cameras. Some examples of PTD are shown in Fig. 2. 

Fig. 1:The spatial distribution of the cameras in the Person Trajectory Dataset. For each camera, the satellite enlarged image and the camera view of the corresponding cameras are displayed. The following numbers, such as SQ0921, indicate the camera index.

![Figure1](https://github.com/zhangxin1995/PTD/blob/main/images/location.jpg)

Fig. 2:Example of Person Trajectory Dataset. The first column is the track of the corresponding person in the satellite image, the second column is the track of the corresponding person in the perspective of the corresponding camera, and the subsequent is the corresponding tracked person image.

![Figure2](https://github.com/zhangxin1995/PTD/blob/main/images/Figure19.jpg)

## Spatial Temporal Dataset
For spatio-temporal data, it consists of a training set and testing set, within which the data are derived from of PTD's training set and testing set, respectively. 

You can download this dataset through the following link: [Download](https://drive.google.com/file/d/1GPSSlPe6ZwFTNAzLOsOk1Li9DdYAvBkG/view?usp=sharing)

You can load the dataset through the following code:
```python
import pickle as pkl
with open('spatial_temporal_dataset.pkl','rb') as infile:
    datas=pkl.load(infile)
```
 The dataset is represented by a `dict`, and each of them has the following meanings:

1. train_set: It has three dimensions. The first dimension represents the camera index `1` and the second dimension represents the camera index `2`. You can get a one-dimensional list showing the walking time from camera index `1` to camera index `2`.
2. train_y: It has the same structure as train_set, except that the list indicates whether the corresponding data is a positive sample. For example, if train_set[1][2][0] is a postive sample, train_y[1][2][0] is 1. If train_set[1][2][0] is a negative sample, train_y[1][2][0] is 0.
3. test_set: It has three dimensions. The first dimension represents the camera index `1` and the second dimension represents the camera index `2`. You can get a one-dimensional list showing the walking time from camera index `1` to camera index `2`.
4. test_y: It has the same structure as train_set, except that the list indicates whether the corresponding data is a positive sample. For example, if test_set[1][2][0] is a postive sample, train_y[1][2][0] is 1. If train_set[1][2][0] is a negative sample, test_y[1][2][0] is 0.
5. distmat: The distance between different camera pairs, for example, datas['distmat'][1][2] represents the distance from camera index `1` to camera index `2`.

The mapping from camera name to index can be expressed as follows:

```
{'SQ0925': 5, 'SQ0924': 8, 'SQ0927': 2, 'SQ0926': 7, 'SQ0921': 6, 'SQ0923': 0, 'SQ0930': 1, 'SQ0922': 4, 'SQ0931': 3}
```

## Trajectory Dataset
You can download this dataset through the following link:
| Feature Model | Link |
| -----| ---- | 
| Resnet50 |[Download](https://drive.google.com/file/d/1-o-CZsyc1IN94bPosgrNyQJhP8GLxtrL/view?usp=sharing) |
| MGN |[Download](https://drive.google.com/file/d/1TxdOX_lh110FlBWlIgUj8tzNC3xgP9jd/view?usp=sharing)  |
| PCB | [Download](https://drive.google.com/file/d/1bZFjVgpsF2qnSKrZn8R67qHUG6XK7bKl/view?usp=sharing) | 
| CAP | [Download](https://drive.google.com/file/d/137bEavdJGL-ID9H5iLq_vjEhQMVeJXwf/view?usp=sharing) |

You can load the dataset through the following code:
```python
import pickle as pkl
with open('resnet50_visual_dataset.pkl','rb') as infile:
    datas=pkl.load(infile)
```
Before introducing the format of the visual dataset, we should first understand several indexes in the process of person retrieval. In the dataset, we have 5-class indexes, which are as follows:
1. Person index. It represents the identity of pedestrians. Everyone has a unique index.
2. Camera index. It represents the camera number, and each camera has a unique index.
3. Camera tracklet index. It represents the index of a tracklet under a specific camera.
4. Global tracklet index. The global index is obtained by splicing the tracklets under all cameras together.
5. Trajectory index. The trajectory index indicates a tracklet belongs to which trajectory.

The dataset are represented by a `dict`, and each of them has the following meanings:
1. qcs: Camera index list of query tracklets.
2. qts: Timestamp list of query tracklets.
3. qfs: Feature list list of query tracklets. 
4. qls: Person index list of query tracklets.
5. fqcs: Camera list index list of query images.
6. tidxs: Camera tracklet index list of gallery tracklets.
7. fqfs: Feature list of query image.
8. fqls: Person index list of query images.
9. tcs: Camera index list of gallery tracklets.
10. gts: Timestamp list of gallery tracklets.
11. tfs: Feature list of  gallery tracklets.
12. tls: Person index list of gallery tracklets.
13. ftcs: Camera index list of gallery images.
14. ftfs: Feature list of gallery images.
15. ftls: Person index list of gallery images.
16. ftidxs: Camera tracklet index list of gallery images.
17. idx2pathidx: Map the person index to the global tracklet index.
18. tpath2index:Map the trajectory index to the global tracklet index.
19. qidxs: Camera tracklet index list of query tracklets.
20. fqidxs: Camera tracklet index list of query images.


## Code
Please put the downloaded data file in the `data/PTD` directory under the code root directory. And execute the following statement:
```
python main.py -yaml ./config/local_crf_resnet.yaml
```

<<<<<<< HEAD
## Citation
```
Xin Zhang, Xiaohua Xie, Jianhuang Lai, Wei-Shi Zheng. Cross-camera Trajectories Help Person Retrieval in a Camera Network. IEEE Transactions on Image Processing (TIP), in press, 2023.
```


=======
>>>>>>> 6eacb147ff53b7f85fb8d59650e1d2c82ff76f73














