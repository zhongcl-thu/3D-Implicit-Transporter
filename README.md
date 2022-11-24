# 3D Implicit Transporter for Temporal Keypoint Discovery

This repository is a PyTorch implementation.

## Datasets
PartNet-Mobility dataset is provided by UMPNET, which can be downloaded from [here](https://ump-net.cs.columbia.edu/download/mobility_dataset.zip).

We utilize the Pybullet simulator and object models from PartNet-Mobility dataset to generate training and test data, which will be released after our paper published. Then, move the downloaded data into 'data' folder.

Our 'data' folder structure is as follows:

```
data
  ├── bullet_multi_joint_train
  │    ├── FoldingChair
  │    ...
  │    ├── Window
  ├── bullet_multi_joint_test
  │    ├── Box
  │    ...  
  │    ├── Window
  
```


## Installation
Make sure that you have all dependencies in place. The simplest way to do so, is to use anaconda.

You can create an anaconda environment called 3d_transporter using
```
conda create --name 3d_transporter python=3.7
conda activate 3d_transporter
```

**Note**: Install python packages according to the CUDA version on your computer:
```
# CUDA >= 11.0
pip install -r requirements_cu11.txt 
pip install torch-scatter==2.0.9
# CUDA < 11.0
pip install -r requirements_cu10.txt 
pip install torch-scatter==2.0.4
```

Next, compile the extension modules.
You can do this via
```
python setup.py build_ext --inplace
```


## Training

If train on single GPU, run:
```
sh exp/train1110/train_single.sh
```

If train on multiple GPUs, modify the values of 'CUDA_VISIBLE_DEVICES' and 'nproc_per_node' in the 'train_multi.sh' according to the number of available GPUs of yours and run:
```
sh exp/train1110/train_multi.sh
```

## Extract and Save Keypoints
For seen data:
```
sh exp/train1110/test_seen.sh save_kpts
```
For unseen data:
```
sh exp/train1110/test_unseen.sh save_kpts
```

## Evaluate
### 1. Perception
Test for seen data:
```
python tools/eval_repeat.py \
--dataset_root data/bullet_multi_joint_test --test_root ${scriptDir}/test_result/seen/ \
 --test_type seen
```
Test for unseen data:

```
python tools/eval_repeat.py \
--dataset_root data/bullet_multi_joint_test --test_root ${scriptDir}/test_result/unseen/ \
 --test_type unseen
```

### 2. Manipulation 

**Note**: TODO.

