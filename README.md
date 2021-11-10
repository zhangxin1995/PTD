[toc]

# Person Trajectory Dataset

## Description
Person Trajectory Dataset comes from paper `Person Retrieval with Trajectory Generation`. This PTD is collected from a camera network of nine cameras from 8 a.m. to 8 p.m. The spatial distribution of the cameras for this dataset is shown in Fig. 1. The person image sequence acquired are then preliminarily labeled by FPN target detection method and DeepSort tracking method, and then the annotation results are corrected manually. The images of any person that only appears under one camera are also eliminated to ensure the persons in PTD have appeared under at least two different cameras. Some example of PTD are shown in Fig. 2. 

Fig. 1:The spatial distribution of the cameras in the Person Trajectory Dataset. For each camera, the satellite enlarged image and the camera view of the corresponding cameras are displayed. The following numbers, such as SQ0921, indicate the camera index.

![Figure1](https://github.com/zhangxin1995/PTD/blob/main/images/location.jpg)

Fig. 2:Example of Person Trajectory Dataset. The first column is the track of the corresponding person in the satellite image, the second column is the track of the corresponding person in the perspective of the corresponding camera, and the subsequent is the corresponding tracked person image.

![Figure2](https://github.com/zhangxin1995/PTD/blob/main/images/Figure19.jpg)

## Spatial Temport Dataset
The Spatio-Temporal dataset contains two parts: the distance between cameras and the time set when a person walks from one camera to another in the data set.

















