import numpy as np
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
import glob

def load_and_normalize_image(path):
    # Load an image using OpenCV and convert it to RGB
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize the image to [0, 1] range
    return image.astype(np.float32) / 255.0

# Paths to denoised images and ground truth
denoised_images_paths = sorted(glob.glob('./results/Real_Denoising/SIDD/png/*.png'))
gt_data = loadmat('./Datasets/test/SIDD/ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb']
gt_data = gt_data.astype(np.float32) / 255.0  # Normalize GT data

# Ensure the number of denoised images matches the expected count
num_images, num_blocks, _, _, _ = gt_data.shape
assert len(denoised_images_paths) == num_images * num_blocks, "Mismatch in the number of blocks"

total_psnr, total_ssim = 0.0, 0.0

for i in range(num_images):
    for k in range(num_blocks):
        # Index for the denoised image path
        index = i * num_blocks + k
        denoised_image_path = denoised_images_paths[index]
        
        # Load and normalize denoised image
        denoised_image = load_and_normalize_image(denoised_image_path)
        
        # Get corresponding GT image
        gt_image = np.squeeze(gt_data[i, k, :, :, :])

        # Compute PSNR and SSIM on the RGB data
        current_psnr = psnr(gt_image, denoised_image, data_range=1.0)
        current_ssim = ssim(gt_image, denoised_image, data_range=1.0, multichannel=True)
        
        total_psnr += current_psnr
        total_ssim += current_ssim

# Calculate and print average PSNR and SSIM
average_psnr = total_psnr / (num_images * num_blocks)
average_ssim = total_ssim / (num_images * num_blocks)
print(f'Average PSNR: {average_psnr:.4f}, Average SSIM: {average_ssim:.4f}')
