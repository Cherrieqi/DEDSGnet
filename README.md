# DEDSGnet

![DEDSGnet](https://github.com/Cherrieqi/DEDSGnet/blob/main/Fig.1.tif)


[A De-stylized Expanded Domain Sample Generation Network for Hyperspectral Image Cross-Domain Classification]



## Requirements

This code is based on **Python 3.10** and **Pytorch 1.12**.

*Installation list:*

**· pytorch**

**· matplotlib**

**· opencv-python**

**· scipy**

**· h5py**

**· tqdm**

**· scikit-learn**


## Models

**· SD--H13+H18 :** [model5.pth](https://pan.baidu.com/s/1BNkfPxjGgY4KI1Ih7i3e6Q?pwd=y2we)

**· SD--PU+PC :** [model5.pth](https://pan.baidu.com/s/1rqzCC1tDEI6Jm_Loyhk5tQ?pwd=cpn9)


## Datasets

**· [raw](https://pan.baidu.com/s/1iDQoBf2sfl6WAXyOXC15FQ?pwd=9azr) :** Houston2013 / Houston2018 / PaviaU / PaviaC

**· [H13+H18--PU/PC](https://pan.baidu.com/s/1FRozdjaxXablec2JdclPUg?pwd=smy2) :** gen_H13 / gen_H18 / gen_PU / gen_PC

**· [PU+PC--H13/H18](https://pan.baidu.com/s/1g0pHClw-um-RRhWcdIDtrQ?pwd=npar) :** gen_PU / gen_PC / gen_H13 / gen_H18 



## Getting start:

##### · Dataset structure

```
data/H1318
├── gen_H13
│   ├── img.npy
│   └── gt.npy
├── gen_H18
│   ├── img.npy
│   └── gt.npy
├── gen_PC
│   ├── img.npy
│   └── gt.npy
└── gen_PU
     ├── img.npy
     └── gt.npy
```


```
data/PUPC
├── gen_PU
│   ├── img.npy
│   └── gt.npy
├── gen_PC
│   ├── img.npy
│   └── gt.npy
├── gen_H13
│   ├── img.npy
│   └── gt.npy
└── gen_H18
     ├── img.npy
     └── gt.npy
```


```     
data/raw
├── Houston2013
│   ├── Houston.mat
│   └── Houston_gt.mat
├── Houston2018
│   ├── HoustonU.mat
│   └── HoustonU_gt.mat
├── PaviaC
│   ├── pavia.mat
│   └── pavia_gt.mat
└── PaviaU
     ├── paviaU.mat
     └── paviaU_gt.mat
```



**NOTE:**

​       Training and test data can be generated via *data_gen_xxxxx.py* respectively. Where *_H1318* indicates that the source domains are H13 and H18 and *_PUPC* indicates that the source domains are PU and PC.

##### · Train

​       Run *train_xxxx.py*. 

##### · Test

​       Run *te_xxxx.py*. 















