# MobileSAM-pytorch

Reproduction of MobileSAM using pytorch ([our reimplemented MobileSAM model weights](https://drive.google.com/drive/folders/1kzdY2GuJM3B8ssZWOBJhqWXx-QaEPT9e?usp=drive_link)). Please star it if it helps you, thank you!

<p align="center">
<img src="vis/223754.jpg?raw=true" width="99.1%" /></br>
<b>From left to right: SAM result, MobileSAM result, our re-implemented MobileSAM result.</b>
</br>
</p>

* Download the MobileSAM encoder we re-trained from [here](https://drive.google.com/drive/folders/1kzdY2GuJM3B8ssZWOBJhqWXx-QaEPT9e?usp=drive_link) for your usage. We provide the whole retrained mobilesam model weights ```retrained_mobilesam.pth```, and the only image encoder part ```image_emcoder.pth``` for your reference.

* Please find visualization results in folder [vis](https://github.com/YuYue525/MobileSAM-distiller/tree/main/vis). We compared [SAM](https://github.com/facebookresearch/segment-anything), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) and our reimplemented version of MobileSAM.

## Installation

### Clone the Repository:

```
git clone --recursive https://github.com/YuYue525/MobileSAM-pytorch.git
cd MobileSAM-pytorch
mv *.py ./MobileSAM
cd MobileSAM; pip install -e .
```

### Environment Preparation

Please use ```conda``` to create an environment and download all the packages required (we use ```python==3.8.13```, ```pytorch==1.13.0+cu117```, ```torchvision==0.14.0+cu117```):

```
pip install -r requirements.txt 
```

## Dataset Preparation

Download SA-1B dataset parts from [here](https://ai.meta.com/datasets/segment-anything/) and unzip them, and then pre-process the dataset using:

```
# dataset downloading in dataset dir
wget -b -c -O "sa_<index>.tar" "<link>"

# unzip the downloaded file
mkdir sa_<index>
tar -xvf sa_<index>.tar -C sa_<index>

# data preprocess: extract features by SAM teacher as "target" and save them as .npy
python preprocess.py --dataset_dir sa_<index>
```
> [!NOTE]
> The purpose of pre-processing is to save the SAM output feature maps to the local hard drive, then there is no need to use the image encoder of the SAM model to repeatedly perform forward inference during training. This method is also mentioned in MobileSAM paper. Therefore, please make sure your hard drive has enough space！(around 500GB for 1% SA-1B after pre-processing)

e.g., after downloading ```sa_000000.tar```, we unzipped the file into the file folder ```sa_000000```, we can run ```python preprocess.py --dataset_dir sa_000000``` to pre-process the data to generate features' ```.npy``` file. In our experiments, we downloaded 1% SA-1B dataset as our training set (from ```sa_000000``` to ```sa_000009```) and another 0.1% as our validation set (```sa_000010```).


## Distillation Process

We can distill the knowledge from SAM to our MobileSAM using the following command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --optim <optimizer> --learning_rate <lr> --weight_decay <wd> --work_dir <work_dir>
```

> [!NOTE]
> Please carefully check all the file paths in the code!

e.g., ```CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --optim adamw --learning_rate 0.001 --weight_decay 0.0005 --batch_size 8 --epochs 16 --work_dir exp/adamw_lr_1e-3_wd_5e-4_bs_8_epoch_16"```

## Model Aggregation

Note that our trained model is only the <b>image encoder</b> part of the whole MobileSAM model. To aggregate the trained encoder with freezed prompt encoder and mask decoder, please run:

```
python model_aggregation.py --ckpt <pth_path> --save_model_path <save_path> --save_model_name <model_name>
```
e.g., ```python model_aggregation.py --ckpt exp/adamw_lr_1e-3_wd_5e-4_bs_8_epoch_16/ckpt/final.pth --save_model_path weights --save_model_name adamw_lr_1e-3_wd_5e-4_bs_8_epoch_16.pth```

## Evaluation

We evaluate the model through segmenting everything and visualize the results by indicating ```--vis```, and evaluate the mIoU compared with SAM results by indicating ```--miou```

```
python eval.py --ckpt <checkpoint_path> --work_dir <work_dir> --vis True --miou True
```

e.g., ```python eval.py --ckpt exp/adamw_lr_1e-3_wd_5e-4_bs_8_epoch_16/ckpt/final.pth --work_dir exp/adamw_lr_1e-3_wd_5e-4_bs_8_epoch_16 --vis True --miou True```

## ONNX Export

Export the model with

```
python scripts/export_onnx_model.py --checkpoint ./weights/mobile_sam.pt --model-type vit_t --output ./mobile_sam.onnx
```

Also check the [example notebook](https://github.com/ChaoningZhang/MobileSAM/blob/master/notebooks/onnx_model_example.ipynb) to follow detailed steps.


## Acknowledgement

<details>
<summary>
<a href="https://github.com/ChaoningZhang/MobileSAM">MobileSAM</a> (Faster Segment Anything) [<b>bib</b>]
</summary>

```bibtex
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
```
</details>

<details>
<summary>
<a href="https://github.com/facebookresearch/segment-anything">SAM</a> (Segment Anything) [<b>bib</b>]
</summary>

```bibtex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
</details>

<details>
<summary>
<a href="https://github.com/microsoft/Cream/tree/main/TinyViT">TinyViT</a> (TinyViT: Fast Pretraining Distillation for Small Vision Transformers) [<b>bib</b>]
</summary>

```bibtex
@InProceedings{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={European conference on computer vision (ECCV)},
  year={2022}
```
</details>
