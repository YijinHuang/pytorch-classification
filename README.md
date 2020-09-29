## Pytorch Classification

- A general, feasible and extensible framework for classification tasks.
- Easy to transfer to your project.



### Features

- Update all hyperparameters in one file
- Training progress monitoring and curve visualization
- Weighted sampling and weighted loss for imbalance dataset
- Kappa calculation for evaluating model on imbalance dataset
- Different learning rate schedulers and warmup support
- Data augmentation
- Multiple GPUs support




### How to use

1. Organize your dataset as following:

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



2. Update your configurations and hyperparameters in `config.py`.



3. Run:

```shell
$ CUDA_VISIBLE_DEVICES=x python main.py
```



4. Monitor your training progress in website [127.0.0.1:6006](127.0.0.1:6006) by running:

```shell
$ tensorborad --logdir=/path/to/your/log --port=6006
```


