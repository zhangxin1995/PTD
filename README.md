[toc]

# Person Trajectory Dataset

## Description
Person Trajectory Dataset comes from paper `Person Retrieval with Trajectory Generation`. This PTD is collected from a camera network of nine cameras from 8 a.m. to 8 p.m. The spatial distribution of the cameras for this dataset is shown in Fig. 1. The person image sequence acquired are then preliminarily labeled by FPN target detection method and DeepSort tracking method, and then the annotation results are corrected manually. The images of any person that only appears under one camera are also eliminated to ensure the persons in PTD have appeared under at least two different cameras. Some example of PTD are shown in Fig. 2. 

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
The dataset are represented by dict, and each of them has the following meanings:

1. train_set:It has three dimensions. The first dimension represents the camera index `1` and the second dimension represents the camera index `2`. Finally, you can get a one-dimensional list showing the walking time from `1` to `2`.
2. train_y:It has the same structure as train_set, except that the list indicates whether the corresponding data is a positive sample. For example, if train_set[1][2][0] is a postive sample, train_y[1][2][0] is 1. If train_set[1][2][0] is a negative sample, train_y[1][2][0] is 0.

3. test_set:It has three dimensions. The first dimension represents the camera index `1` and the second dimension represents the camera index `2`. Finally, you can get a one-dimensional list showing the walking time from `1` to `2`.
4. test_y:It has the same structure as train_set, except that the list indicates whether the corresponding data is a positive sample. For example, if test_set[1][2][0] is a postive sample, train_y[1][2][0] is 1. If train_set[1][2][0] is a negative sample, test_y[1][2][0] is 0.
5. distmat: The distance between different camera pairs, for example, datas['distmat'][1][2] represents the distance from `1` to `2`.

The mapping from camera name to index can be expressed as follows:

```
{'SQ0925': 5, 'SQ0924': 8, 'SQ0927': 2, 'SQ0926': 7, 'SQ0921': 6, 'SQ0923': 0, 'SQ0930': 1, 'SQ0922': 4, 'SQ0931': 3}
```

# Trajectory Dataset
You can download this dataset through the following link: [Download](https://drive.google.com/file/d/1GPSSlPe6ZwFTNAzLOsOk1Li9DdYAvBkG/view?usp=sharing)

You can load the dataset through the following code:
```python
import pickle as pkl
with open('visual_dataset.pkl','rb') as infile:
    datas=pkl.load(infile)
```
Before introducing the format of visual dataset, we should first understand several indexes in the process of person retrieval.In the dataset, we have 5-class indexes, which are as follows:
1. Person index. It represents the identity of pedestrians. Everyone has an unique index.
2. Camera index. It represents the camera number, and each camera has an unique index.
3. Camera tracklet index. It represents the index of a tracklet under a specific camera.
4. Global tracklet index. The global index is obtained by splicing the tracklets under all cameras together.
5. Trajectory index. The trajectory index indicates a tracklet belongs to which trajectory.

The dataset are represented by dict, and each of them has the following meanings:
1. qcs
2. qts
3. qfs
4. qls
5. fqcs
6. fqts
7. fqfs
8. fqls
9. tcs
10. ts
11. gfs
12. gls
13. fgcs
14. fgts
15. fgfs
16. fgls


















