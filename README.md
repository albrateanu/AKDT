# AKDT: Adaptive Kernel Dilation Transformer for Effective Image Denoising

## Updates

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
