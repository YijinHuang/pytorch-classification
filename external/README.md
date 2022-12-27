# External Tools

#### 1. crop.py

Preprocessing code for EyePACS dataset. This code remove the black border of fundus images and resize them to 512 x 512.
```shell
python crop.py -n 8 --crop-size 512 --image-folder <path/to/image/dataset> --output-folder <path/to/processed/dataset>
```
Here, `-n` is the number of workers. The processed dataset will be saved in the `--output-folder`.