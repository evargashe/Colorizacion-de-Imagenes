
## Prerequisites
* [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-update2)
* Python3
* Pytorch >= 1.5
* Ultralytics (Yolov8)
* OpenCV-Python
* Pillow/scikit-image
* Please refer to the [env.yml](env.yml) for detail dependencies.

## Getting Started
1. Clone this repo:
```sh
git clone https://github.com/evargashe/Colorizacion-de-Imagenes.git
cd Colorizacion-de-Imagenes
```
2. Install [conda](https://www.anaconda.com/).
3. Install all the dependencies
```sh
conda env create --file env.yml
```
4. Switch to the conda environment
```sh
conda activate instacolorization
```
5. Install other dependencies
```sh
sh scripts/install.sh
```

## Pretrained Model
1. Download it from [google drive](https://drive.google.com/open?id=1Xb-DKAA9ibCVLqm8teKd1MWk6imjwTBh).
```sh
sh scripts/download_model.sh
```
2. Now the pretrained models would place in [checkpoints](checkpoints).

## Instance Prediction
Please follow the command below to predict all the bounding boxes fo the images in `example` folder.
```
python inference_bbox.py --test_img_dir example
```
All the prediction results would save in `example_bbox` folder.

## Colorize Images
Please follow the command below to colorize all the images in `example` foler.
```
python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 128 --test_img_dir example --results_img_dir results
```
All the colorized results would save in `results` folder.

* Note: all the images would convert into L channel to colorize in [test_fusion.py's L51](test_fusion.py#L51)

## Training the Model
Please follow this [tutorial](README_TRAIN.md) to train the colorization model.

## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details. 

## Citation Paper
```
@inproceedings{Su-CVPR-2020,
  author = {Su, Jheng-Wei and Chu, Hung-Kuo and Huang, Jia-Bin},
  title = {Instance-aware Image Colorization},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```


