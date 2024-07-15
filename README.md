# GraphSlot

## Introduction

​	Code release for paper : Physics Understanding Enhancement via an Object-centric Learning Method Nested by Graph Representation

​	This code contains:

- Training GraphSlot model on MOVi-C dataset
- Evaluate GraphSlot model on MOVi-C dataset
- Checkpoints of GraphSlot on MOVi-C dataset

![figure](https://github.com/HaronW/Phy-GraphSlot/blob/main/figure.png)



## Installation

​	To setup conda environment, run:

```shell
cd ./PhyGraphSlot
conda env create -f environment.yml
```



## Experiments

#### Data preparation

​	Details about MOVi dataset can be found at [MOVi](https://github.com/google-research/kubric/blob/main/challenges/movi/README.md). The MOVi-C datasets are stored in a [Google Cloud Storage (GCS) bucket](https://console.cloud.google.com/storage/browser/kubric-public/tfds/movi_c) and can be downloaded to local disk prior to training by running:

```shell
cd ./PhyGraphSlot/phygraphslot/datasets
gsutil -m cp -r "gs://kubric-public/tfds/movi_c/128x128" .
```



#### Train

​	To train Phy-GraphSlot, run:

```shell
cd ./PhyGraphSlot
python -m phygraphslot.main --seed 42 --gpu 0,1,2,3 --mode=phygraphslot
```



#### Evaluate checkpoints

​	To evaluate Phy-GraphSlot, run:

```shell
cd ./PhyGraphSlot
python -m phygraphslot.main --seed 42 --gpu 0,1,2,3 --mode=phygraphslot --eval --resume_from ./model/phygraphslot_100000.pt
```



#### Checkpoint

​	Checkpoint of GraphSlot on MOVi-C dataset with 128 x 128 resolution is available at [Google Drive](https://drive.google.com/file/d/1ZVT0aMLixII3F7SMeER_dLw_RT8A4UYx/view?usp=sharing).



## Acknowledgement

​	We thank the authors of [Slot-Attention](https://github.com/google-research/google-research/tree/master/slot_attention), [SAVi](https://github.com/google-research/slot-attention-video/), and [SAVi-PyTorch](https://github.com/junkeun-yi/SAVi-pytorch) for opening source their wonderful works.



## License

​	GraphSlot is released under the MIT License. See the LICENSE file for more details.
