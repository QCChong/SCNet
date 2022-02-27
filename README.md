## SCNet: Point Cloud Completion by shape-aware convolution

Xiangyang Wu, Chongchong Qu, Haixin Zhou, Yongwei Miao

-----------

This repository contains the source code for the paper “SCNet: Point Cloud Completion by shape-aware convolution”.

-----


### Usage

#### 1) Envrionment

- CUDA 11.1
- Python 3.7
- Pytorch 1.9.0
- Pytorch-lightning 1.3.8
- [knn_cuda](https://github.com/unlimblue/KNN_CUDA/releases/tag/0.2) 0.2

#### 2) Compile

Compile loss and  PyTorch 3rd-party modules

```bash
    cd loss/emd
    python setup.py install

    cd utils/pointnet2_ops_lib
    python setup.py install
```

#### 3) Download data [(MVP dataset)](https://www.dropbox.com/sh/la0kwlqx4n2s5e3/AACjoTzt-_vlX6OF9mfSpFMra?dl=0&lst=)
```bash
    cd data
    bash get_dataset.sh
```

#### 4) Train or validation

1. train the model

   ```bash
   python train.py 
   ```

2. validate the model. [Here](https://drive.google.com/file/d/1P2JvCiHxkTt7WxbwC70L0nThdT0Z6GWo/view?usp=sharing) is a pretrained model. Download them here, and unzip it under logs/.

   ```bash
   python evalution.py
   ```

## Acknowledgement

This code is based on  [VRCNet](https://github.com/paul007pl/VRCNet) and [Pointnet2.Pytorch](https://github.com/erikwijmans/Pointnet2_PyTorch).
