# CoronaryArteryStenosisLevelClassification
We utilize DNNs for identifying the level of stenosis in coronary arteries from CT scans and MPR images

It is the implementation of the CNN-CASS: CNN for Classification of Coronary Artery Stenosis Score inMPR Images. 

Maria Dobko, Bohdan Petryshak, Oles Dobosevych

In CVWW

## Overview

## General Pipeline

## Usage

## Dataset structure

The dataset has the following structure:

```
|-- RootDir
|     |-- Train
|           |-- labels.csv/.xlsx
|           |-- imgs
|                 |--  patient_1
|                       |-- LAD
|                       |-- RCA
|                       ...
|                 ...
|     |-- Val
|           |-- labels.csv/.xlsx
|           |-- imgs
|                 |--  val_patient_1
|                       |-- LAD
|                       |-- RCA
|                       ...
|                 ...
|     |-- Test
|           |-- labels.csv/.xlsx
|           |-- imgs
|                 |--  test_patient_1
|                       |-- LAD
|                       |-- RCA
|                       ...
|                 ...
|
```

## Config
Our config organized as follows:

1. `experiments_path` - relative or absolute path to output of the TensorBoard.
```
experiments_path: experiments_lenet/

```

2. `device` - device, where to run on. 
```
device: cuda
```

3. `random_state` - fix random seed.
```
random_state: 42
```

4. `dataloader` - parameters for the dataloader class: 
- **batch_size** - size of the batch
- **sampler** - sampler, which balances the classes within epoch(ImbalancedDatasetSampler), or the batch(BalancedBatchSampler) level.
- **accumulation_steps** - amount of the gradient accumulation steps.
```
dataloader:
  batch_size: 16
  sampler: BalancedBatchSampler
  accumulation_steps: 1
```

5. `data` - parameters for the input data:
- **root_dir:** - relative or absolute path to the dataset
- **filters:**
  - **arteries** - arteries, which we want to include into the training process.
  - **viewpoint_index_step** MPR viewpoint index step.
- **groups:**
  - **class_index:** [list of the stenosis score categories, which belongs to this class index]
  - ...
- **augmentation:**
  - **name:** - name of the class, which we want to use for augmentation(LightAug, MediumAugFixed, StrongAugFixed, SafeAug...)
  - **parameters:**
    - **p** - probability of implying the random augmentation techniques for the input images
- **dataset** - dataset class, used for loading the input data(MPR_Dataset, MPR_Dataset_New_Test, MPR_Dataset_LSTM...)
```
data:
  root_dir: data/all_branches_with_pda_plv_with_new_test
  filters:
    arteries: [ 'LAD', 'D-1', 'D-2', 'D-3','PLV_RCA', 'LCX',  'OM-2', 'RCA', 'PLV_LCX', 'OM-3', 'PDA_LCX','OM-1', 'OM' ]
    viewpoint_index_step: 1
  groups:
    0: [ 'NORMAL','-']
    1: ['250%', '<25%', '<35%', '25-50%', '25%', '<50%']
    2: ['50%', '70%','*50%', '50-70%', '70-90%', '90-100%', '>50%', '>70%', '90%', '>90%','75%', '>75%']
  augmentation:
    name: MediumAugFixed
    parameters:
      p: 0.8
  dataset: MPR_Dataset
```

6. `model` - choose the model, which you want for training:
- **name** - name of the model(ShuffleNetv2, ResNet50, ResNet18EfficientB4, EfficientB4, LSTMDeepResNetClassification, AttentionResNet34...)
- **parameters:**
  - **pretrained** - wether backbone pretrained on ImageNet. If you use custom architecture, comment the parameters and pretrained lines.(we will show in the example below)
```
model:
  name: ShuffleNetv2
#  parameters:
#    pretrained: True
```

7. `optimizer` - parameters for the optimizer:
- **name** - name of the optimizer class.(use names of the standart optimizers in PyTorch)
- **parameters:**
  - **lr** - learning rate.
  - **weight_decay** - size of the momentum decay.(uncomment this parameter if needed)
  - **momentum** - momentum size.(uncomment this parameter if needed)
```
  optimizer:  # as in paper
  name:  Adam
  parameters:
    lr: 0.0001
#    weight_decay: 0.00001
#    momentum: 0.9
```
8. `loss` - parameters for the loss function
- **name** - name of the loss class(CrossEntropyLabelSmooth, CrossEntropyLoss, OHEMLoss)
- **parameters:**
    - **num_classes** - number of classes
    - **k** - parameter for OHEMLoss(comment for other classes), top-K elements, the most hard cases in the batch.
```
loss:
  name: CrossEntropyLabelSmooth #CrossEntropyLabelSmooth #CrossEntropyLoss OHEMLoss
  parameters:
     num_classes: 3
#    k: 40
```


## Citation

If you use this code for your research, please cite our paper.

```
​```
@InProceedings{DobkoPetryshakDobosevych_2020,
author = {Maria Dobko and Bohdan Petryshak and Oles Dobosevych},
title = {CNN-CASS: CNN for Classification of Coronary Artery Stenosis Score inMPR Images},
booktitle = {The Computer Vision Winter Workshop (CVWW)},
month = {Feb},
year = {2020}
}
​```
```
<p style='color:red'>This is some red text.</p>
