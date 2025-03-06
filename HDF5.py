import h5py
import tifffile as tiff
import numpy as np
import os
from multiprocessing import Pool, cpu_count

# Directories
image_dir = r'K:\21012025_BSF obj x10\C8\Long term\best LUT'
output_dir_large = r'K:\21012025_BSF obj x10\C8\Long term\HDF5\Large_HDF5'
output_dir_cropped = r'K:\21012025_BSF obj x10\C8\Long term\HDF5\Cropped_HDF5'

# Parameters for the 5D structure
time_points = 1  # Set to 1 for static images
z_slices = 1  # Set to 1 if not using Z-stacks
channel_count = 3  # Set based on image type (3 for RGB, 1 for grayscale)
crop_size = 7000  # Size of the cropped image

# Ensure the output directories exist
os.makedirs(output_dir_large, exist_ok=True)
os.makedirs(output_dir_cropped, exist_ok=True)

def rename_image_file(img_name):
    original_file_path = os.path.join(image_dir, img_name)
    new_file_path = original_file_path.replace('μ', 'u')
    os.rename(original_file_path, new_file_path)
    return os.path.basename(new_file_path)

def process_image(img_name):
    img_name = rename_image_file(img_name)  # Rename the file
    img_path = os.path.join(image_dir, img_name)
    img = tiff.imread(img_path)  # Load with original data

    # Convert the image to a 5D numpy array
    images_np = np.array([img])
    n_images, height, width, channels = images_np.shape
    images_5d = images_np.reshape((time_points, z_slices, height, width, channel_count))

    # Define the output HDF5 file paths
    output_hdf5_file_large = os.path.join(output_dir_large, f"{os.path.splitext(img_name)[0]}.h5")
    output_hdf5_file_cropped = os.path.join(output_dir_cropped, f"{os.path.splitext(img_name)[0]}_cropped.h5")

    # Save the full image to HDF5 file
    with h5py.File(output_hdf5_file_large, 'w') as f:
        f.create_dataset(
            'tzyxc_images',
            data=images_5d,
            compression="gzip",
            compression_opts=9,
            dtype=img.dtype  # Preserve original data type
        )

    center_y, center_x = height // 2, width // 2
    start_y, start_x = center_y - crop_size // 2, center_x - crop_size // 2
    cropped_img = img[start_y:start_y + crop_size, start_x:start_x + crop_size]

    # Convert the cropped image to a 5D numpy array
    cropped_images_np = np.array([cropped_img])
    cropped_images_5d = cropped_images_np.reshape((time_points, z_slices, crop_size, crop_size, channel_count))

    # Save the cropped image to HDF5 file
    with h5py.File(output_hdf5_file_cropped, 'w') as f:
        f.create_dataset(
            'tzyxc_images',
            data=cropped_images_5d,
            compression="gzip",
            compression_opts=9,
            dtype=img.dtype  # Preserve original data type
        )

    print(f"Image {img_name} successfully processed and saved in {output_hdf5_file_large} and {output_hdf5_file_cropped}")

if __name__ == "__main__":
    img_names = [img_name for img_name in os.listdir(image_dir) if img_name.endswith('.tif')]
    with Pool(cpu_count()) as pool:
        pool.map(process_image, img_names)