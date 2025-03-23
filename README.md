# AKDT: Adaptive Kernel Dilation Transformer for Effective Image Denoising

<div align="center">

[![arXiv](https://img.shields.io/badge/Paper-PDF-179bd3)](https://www.scitepress.org/Papers/2025/131577/131577.pdf)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/akdt-adaptive-kernel-dilation-transformer-for/color-image-denoising-on-cbsd68-sigma15)](https://paperswithcode.com/sota/color-image-denoising-on-cbsd68-sigma15?p=akdt-adaptive-kernel-dilation-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/akdt-adaptive-kernel-dilation-transformer-for/color-image-denoising-on-cbsd68-sigma25)](https://paperswithcode.com/sota/color-image-denoising-on-cbsd68-sigma25?p=akdt-adaptive-kernel-dilation-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/akdt-adaptive-kernel-dilation-transformer-for/color-image-denoising-on-mcmaster-sigma15)](https://paperswithcode.com/sota/color-image-denoising-on-mcmaster-sigma15?p=akdt-adaptive-kernel-dilation-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/akdt-adaptive-kernel-dilation-transformer-for/color-image-denoising-on-mcmaster-sigma25)](https://paperswithcode.com/sota/color-image-denoising-on-mcmaster-sigma25?p=akdt-adaptive-kernel-dilation-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/akdt-adaptive-kernel-dilation-transformer-for/color-image-denoising-on-mcmaster-sigma50)](https://paperswithcode.com/sota/color-image-denoising-on-mcmaster-sigma50?p=akdt-adaptive-kernel-dilation-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/akdt-adaptive-kernel-dilation-transformer-for/color-image-denoising-on-urban100-sigma15-1)](https://paperswithcode.com/sota/color-image-denoising-on-urban100-sigma15-1?p=akdt-adaptive-kernel-dilation-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/akdt-adaptive-kernel-dilation-transformer-for/image-denoising-on-urban100-sigma15)](https://paperswithcode.com/sota/image-denoising-on-urban100-sigma15?p=akdt-adaptive-kernel-dilation-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/akdt-adaptive-kernel-dilation-transformer-for/color-image-denoising-on-urban100-sigma25)](https://paperswithcode.com/sota/color-image-denoising-on-urban100-sigma25?p=akdt-adaptive-kernel-dilation-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/akdt-adaptive-kernel-dilation-transformer-for/image-denoising-on-urban100-sigma50)](https://paperswithcode.com/sota/image-denoising-on-urban100-sigma50?p=akdt-adaptive-kernel-dilation-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/akdt-adaptive-kernel-dilation-transformer-for/color-image-denoising-on-cbsd68-sigma50)](https://paperswithcode.com/sota/color-image-denoising-on-cbsd68-sigma50?p=akdt-adaptive-kernel-dilation-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/akdt-adaptive-kernel-dilation-transformer-for/color-image-denoising-on-urban100-sigma50)](https://paperswithcode.com/sota/color-image-denoising-on-urban100-sigma50?p=akdt-adaptive-kernel-dilation-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/akdt-adaptive-kernel-dilation-transformer-for/image-denoising-on-sidd)](https://paperswithcode.com/sota/image-denoising-on-sidd?p=akdt-adaptive-kernel-dilation-transformer-for)

</div>

## Updates
- `23.03.2025` Repository updated with link to paper PDF.
- `04.12.2024` Paper has been accepted at VISAPP 2025. To be published.
## Experiment

### 1. Create Environment
- Make Conda Environment
```bash
conda create -n AKDT python=3.7
conda activate AKDT
```
- Install Dependencies
```bash
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```
- Install basicsr
```bash
python setup.py develop --no_cuda_ext
```

### 2. Prepare Datasets
Download the datasets and place them as specified in the ```./Denoising/Datasets/README.md```

SIDD_train - [Google Drive](https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view?usp=sharing)

SIDD_val - [Google Drive](https://drive.google.com/file/d/1Fw6Ey1R-nCHN9WEpxv0MnMqxij-ECQYJ/view?usp=sharing)

SIDD_test - [Google Drive](https://drive.google.com/file/d/11vfqV-lqousZTuAit1Qkqghiv_taY0KZ/view?usp=sharing)

BSD400 - [Google Drive](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing)

DIV2K - [Google Drive](https://drive.google.com/file/d/13wLWWXvFkuYYVZMMAYiMVdSA7iVEf2fM/view?usp=sharing)

WaterlooED - [Google Drive](https://drive.google.com/file/d/19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr/view?usp=sharing)

gaussian_test - [Google Drive](https://drive.google.com/file/d/1mwMLt-niNqcQpfN_ZduG9j4k6P_ZkOl0/view?usp=sharing)

### 3. Test
Pre-trained weights available at [Google Drive](https://drive.google.com/drive/folders/1n6hCeLXxJ2IDtSufdDB0dLo-gSLCHaRp?usp=sharing). Place the pre-trained weights into the ```./Denoising/pretrained_models/``` directory.

- Real Image Denoising evaluation
```bash
# To obtain denoised results
python test_real_denoising_sidd.py --save_images
# Compute PSNR
python eval_sidd.py
```

- Color Gaussian Image Denoising evaluation
```bash
# To obtain denoised results
python test_gaussian_color_denoising.py --model_type blind --sigmas 15,25,50
# Compute PSNR
python evaluate_gaussian_color_denoising.py --model_type blind --sigmas 15,25,50
```

**Note**: ```--weights``` argument can be used to specify paths to different weights.

### 4. Compute Complexity
You can test the model complexity (FLOPS/MACs/Params) using the following command:
```bash
python ./basicsr/models/archs/macs.py
```

### 5. Train
- Generate training image patches:
```bash
# Gaussian color image denoising
python generate_patches_dfwb.py 
# Real image denoising
python generate_patches_sidd.py 
```

- Train AKDT on Color Gaussian Image Denoising:
```bash
./train.sh Denoising/Options/GaussianColorDenoising_AKDT.yml
```

- Train AKDT on Real Image Denoising:
```bash
./train.sh Denoising/Options/GaussianColorDenoising_AKDT.yml
```


## Citation
```
@inproceedings{brateanu2025akdt,
  author    = {Alexandru Brateanu and Raul Balmez and Adrian Avram and Ciprian Orhei},
  title     = {AKDT: Adaptive Kernel Dilation Transformer for Effective Image Denoising},
  booktitle = {Proceedings of the 20th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 3: VISAPP},
  year      = {2025},
  pages     = {418--425},
  isbn      = {978-989-758-728-3},
  issn      = {2184-4321}
}

```
