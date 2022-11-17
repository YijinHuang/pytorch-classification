# Pytorch Classification

- A general, feasible and extensible framework for 2D image classification.



## Features

- Easy to configure (model, hyperparameters)
- Training progress monitoring and visualization
- Weighted sampling / weighted loss / kappa loss / focal loss for imbalance dataset
- Kappa metric for evaluating model on imbalance dataset
- Different learning rate schedulers and warmup support
- Data augmentation
- Multiple GPUs support (DDP mode)
- ViT support



## Installation

Recommended environment:
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
$ conda create -n pycls python=3.8
$ conda activate pycls
$ pip install -r requirements.txt
```



## How to use

**1. Use one of the following two methods to build your dataset:**

- Folder-form dataset:

Organize your images as follows:

```
├── your_data_dir
    ├── train
        ├── class1
            ├── image1.jpg
            ├── image2.jpg
            ├── ...
        ├── class2
            ├── image3.jpg
            ├── image4.jpg
            ├── ...
        ├── class3
        ├── ...
    ├── val
    ├── test
```

Here, `val` and `test` directory have the same structure of  `train`.  Then replace the value of 'data_path' in BASIC_CONFIG in `configs/default.yaml` with path to your_data_dir and keep 'data_index' as null.

- Dict-form dataset:

Define a dict as follows:

```python
your_data_dict = {
    'train': [
        ('path/to/image1', 0), # use int. to represent the class of images (start from 0)
        ('path/to/image2', 0),
        ('path/to/image3', 1),
        ('path/to/image4', 2),
        ...
    ],
    'test': [
        ('path/to/image5', 0),
        ...
    ],
    'val': [
        ('path/to/image6', 0),
        ...
    ]
}
```

Then use pickle to save it:

```python
import pickle
pickle.dump(your_data_dict, open('path/to/pickle/file', 'wb'))
```

Finally, replace the value of 'data_index' in BASIC_CONFIG in `configs/default.yaml` with 'path/to/pickle/file' and set 'data_path' as null.

**2. Update your training configurations and hyperparameters in `configs/default.yaml`.**

**3. Run to train:**

```shell
$ CUDA_VISIBLE_DEVICES=x python main.py
```

Optional arguments:
```
-c yaml_file      Specify the config file (default: configs/default.yaml)
-o                Overwrite save_path and log_path without warning
-p                Print configs before training
```

**4. Monitor your training progress in website [127.0.0.1:6006](127.0.0.1:6006) by running:**

```shell
$ tensorborad --logdir=/path/to/your/log --port=6006
```

[Tips to use tensorboard on a remote server](https://blog.yyliu.net/remote-tensorboard/)



## External Tools

### Diabetic Retinopathy Detection

The codes for the following paper are integrated into this repository.

> Huang, Y., Lin, L., Cheng, P., Lyu, J. and Tang, X., 2021. Identifying the key components in ResNet-50 for diabetic retinopathy grading from fundus images: a systematic investigation. *arXiv preprint arXiv:2110.14160*. [[link](https://arxiv.org/abs/2110.14160)]

### Usage

1. Download EyePACS dataset [[link](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)] and organize the folder to Folder-form as mention above. Then use `external/crop.py` to remove the black border of images and resize them to 512 x 512.

2. Update the dataset path in the `configs/eyepacs.yaml`. Then running the code using it as the configuration file:

```shell
$ CUDA_VISIBLE_DEVICES=x python main.py -c ./configs/eyepacs.yaml
```

3. The features from the trained model are fed into the pair fusion part. The code for the pair fusion will be updated soon.

### Acknowledge

We greatly thanks the reviews of MIDL 2022 for improving this work.



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

