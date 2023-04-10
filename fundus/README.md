# Diabetic Retinopathy

We provide a strong, standardized, and scalable CNN baseline for DR grading. Details can be found [here](https://arxiv.org/abs/2110.14160). 


## Recommended environment for reproduction
To reproduce the experimental results in the paper, the following environments are recommended.

- python 3.8
- pytorch 1.7.1
- torchvision 0.8.2
- timm 0.5.4
- tqdm
- munch
- packaging
- tensorboard

To install the dependencies, run:
```shell
$ git clone https://github.com/YijinHuang/pytorch-classification.git
$ cd pytorch-classification
$ conda create -n fundus python=3.8
$ conda activate fundus
$ pip install -r fundus_requirements.txt
```


## Usage

1. Download EyePACS dataset [[link](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)] and organize the folder to Folder-form as mention above.

2. Preprocessing code for EyePACS dataset. This code remove the black border of fundus images and resize them to 512 x 512.

```shell
python crop.py -n 8 --crop-size 512 --image-folder <path/to/image/dataset> --output-folder <path/to/processed/dataset>
```
Here, `-n` is the number of workers. The processed dataset will be saved in the `--output-folder`.

3. Update the dataset path in the `~/configs/eyepacs.yaml`. Then running the code using it as the configuration file:

```shell
$ CUDA_VISIBLE_DEVICES=x python main.py -c ./configs/eyepacs.yaml
```

4. The features from the trained model are fed into the pair fusion part. The code for the pair fusion will be updated soon.


## Evaluate pre-trained models
This training setting is also adopted for evaluation of pretrained models in [Lesion-based CL](https://arxiv.org/pdf/2107.08274.pdf). To fine-tune pretrained models, treat the pretrained weights as checkpoints by updating the item "checkpoint" in `~/configs/eyepacs.yaml`.


## Acknowledge

We greatly thanks the reviewers of this paper for improving this work.


## Citation

If you find this repository useful, please cite the paper: 

```
@article{huang2021identifying,
  title={Identifying the key components in ResNet-50 for diabetic retinopathy grading from fundus images: a systematic investigation},
  author={Huang, Yijin and Lin, Li and Cheng, Pujin and Lyu, Junyan and Tang, Xiaoying},
  journal={arXiv preprint arXiv:2110.14160},
  year={2021}
}
```