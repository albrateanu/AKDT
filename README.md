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
Download the LOLv1 and LOLv2 datasets:

SIDD_train - [Google Drive](https://drive.google.com/file/d/1vhJg75hIpYvsmryyaxdygAWeHuiY_HWu/view?usp=sharing)

SIDD_val - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)

SIDD_test - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)

BSD400 - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)

DIV2K - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)

WaterlooED - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)

gaussian_test - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)

### 3. Test
Pre-trained weights

- SIDD evaluation
```bash
# To obtain denoised results
python test_real_denoising_sidd.py --save_images
# Compute PSNR
python eval_sidd.py
```

### 4. Compute Complexity
You can test the model complexity (FLOPS/MACs/Params) using the following command:
```bash
python ./basicsr/models/archs/macs.py
```

### 5. Train
- Generate training image patches:
```bash
python generate_patches_dfwb.py 
```

- Train AKDT on color image denoising:
```bash
# Linux
./train.sh Denoising/Options/GaussianColorDenoising_Restormer.yml
# Windows
./train.ps1 Denoising/Options/GaussianColorDenoising_Restormer.yml
```

- Train SIDD on real image denoising:
```bash
# Generate training image patches
python generate_patches_sidd.py 
# Linux
./train.sh Denoising/Options/GaussianColorDenoising_Restormer.yml
# Windows
./train.ps1 Denoising/Options/GaussianColorDenoising_Restormer.yml
```

## Citation
