## Code for Classification

- Implemented in Pytorch.

- Can be easily applied in simple classification tasks.



### Available Models

- ResNet, Inception or other models in torchvision (you can add them in `config.py`)
- EfficientNet (Use [this](https://github.com/lukemelas/EfficientNet-PyTorch) implementation)



### Tricks Implemented

- Adaptive data sampling weight
- Warm up
- Cosine decay
- Affine data augmentation
- Focal loss
- Angular loss
- Kappa prior



### How to Use

#### Organize your dataset

Your should organize your dataset as following:

```
├── your_data_dir
	├── train
		├── class1
			├── image.jpg
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

Here, `val` and `test` directory have the same structure of  `train`.  



#### Run

Most of hyperparameters and configurations are in  `config.py`. 

Change dataset path and network in `BASE_CONFIG` in `config.py`.

Then run:

```shell
$ CUDA_VISIBLE_DEVICES=x python main.py
```

