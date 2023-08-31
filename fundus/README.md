# Diabetic Retinopathy

We provide a strong, standardized, and scalable CNN baseline for DR grading. Details can be found [here](https://www.mdpi.com/2075-4418/13/10/1664). 

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

### Training with single-eye images
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

### Training with paired-eye images. (Optional)
In the EyePACS dataset, both the left and right eyes of a patient are provided. Concatenating the feature vectors of both eyes for classification can significantly improve the performance of DR grading by utilizing the correlation between the two eyes. Please refer the paper for more details.

1. To run the pair fusion, a model trained with single-eye images is required.

2. Define a dict as follows and save as a pkl file (Dict-form dataset).

```python
your_data_dict = {
    'train': [
        # the last item is the DR grade of the first image.
        ('/path/to/image1_left', '/path/to/image1_right', 'DR grade of image1_left'),
        ('/path/to/image1_right', '/path/to/image1_left', 'DR grade of image1_right'),
        ('/path/to/image2_left', '/path/to/image2_right', 'DR grade of image2_left'),
        ('/path/to/image2_right', '/path/to/image2_left', 'DR grade of image2_right'),
        ...
    ],
    'test': [
        ('/path/to/image3_left', '/path/to/image3_right', 'DR grade of image3_left'),
        ('/path/to/image3_right', '/path/to/image3_left', 'DR grade of image3_right'),
        ...
    ],
    'val': [
        ('/path/to/image4_left', '/path/to/image4_right', 'DR grade of image4_left'),
        ('/path/to/image4_right', '/path/to/image4_left', 'DR grade of image4_right'),
        ...
    ]
}

import pickle
pickle.dump(your_data_dict, open('/path/to/pickle/file', 'wb'))
```

3. Run to train
```shell
$ CUDA_VISIBLE_DEVICES=x python fusion.py --data-index /path/to/pickle/file --encoder /path/to/trained_single_eye_model
```


## Evaluate pre-trained models
This training setting is also adopted for evaluation of pretrained models in [Lesion-based CL](https://arxiv.org/pdf/2107.08274.pdf). To fine-tune pretrained models, treat the pretrained weights as checkpoints by updating the item "checkpoint" in `~/configs/eyepacs.yaml`.


## Acknowledge

We greatly thanks the reviewers of this paper for improving this work.


## Citation

If you find this repository useful, please cite the paper: 

```
@article{huang2023identifying,
  title={Identifying the key components in ResNet-50 for diabetic retinopathy grading from fundus images: a systematic investigation},
  author={Huang, Yijin and Lin, Li and Cheng, Pujin and Lyu, Junyan and Tam, Roger and Tang, Xiaoying},
  journal={Diagnostics},
  volume={13},
  number={10},
  pages={1664},
  year={2023},
  publisher={MDPI}
}
```