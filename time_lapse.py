import cv2
import os
from PIL import Image
import zipfile
import re
from moviepy import ImageSequenceClip
import roifile
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

Image.MAX_IMAGE_PIXELS = None
base_path = r'K:\21012025_BSF obj x10'
roi_base_path = r'K:\21012025_BSF obj x10'
chips = [f'C{i}' for i in range(6, 9)]
df = pd.read_csv(r'K:\21012025_BSF obj x10\merged_bacteria_counts_filled.csv', encoding='ISO-8859-1')

def read_roi_file_from_zip(zip_file, roi_file):
    with zip_file.open(roi_file) as file:
        roi_data = file.read()
        roi = roifile.ImagejRoi.frombytes(roi_data)
        coords = roi.coordinates()
    print('ROI file read successfully.')
    return coords

def get_image_paths(input_folder_images):
    images = [f for f in os.listdir(input_folder_images) if f.endswith('.tif')]
    images.sort(key=lambda x: int(re.search(r'T=(\d+)', x).group(1)))
    print('images sorted')
    return [os.path.join(input_folder_images, f) for f in images]

def get_roi_files(zip_ref, arr):
    roi_files = [f for f in zip_ref.namelist() if f.endswith('.roi') and f in arr]
    roi_files.sort()
    print('roi files sorted')
    return roi_files

def get_roi_files_first(zip_ref):
    roi_files = [f for f in zip_ref.namelist() if f.endswith('.roi')]
    roi_files.sort()
    return roi_files

def process_image(image_path, coords, x, y, w, h, droplet_name):
    with Image.open(image_path) as img:
        cropped_img = img.crop((x, y, x + w, y + h))
        cropped_img = np.array(cropped_img)
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    adjusted_coords = [(int(coord[0] - x), int(coord[1] - y)) for coord in coords]
    for i in range(len(adjusted_coords) - 1):
        cv2.line(cropped_img, adjusted_coords[i], adjusted_coords[i + 1], (0, 255, 0), 1)  # Green color, thickness 1
    cv2.line(cropped_img, adjusted_coords[-1], adjusted_coords[0], (0, 255, 0), 1)  # Close the polygon
    # Resize the image
    height, width = cropped_img.shape[:2]
    max_dim = max(height, width)
    scale = min(1280 / width, 720 / height)  # Adjust the scale to fit within 1280x720
    resized_img = cv2.resize(cropped_img, (int(width * scale), int(height * scale)))
    image_name = os.path.basename(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White color
    thickness = 2
    position_name = (10, 30)  # Adjust position for better visibility
    text = f"{image_name.split('.')[0]} - {droplet_name}"
    cv2.putText(resized_img, text, position_name, font, font_scale, color, thickness, cv2.LINE_AA)

    # Create a new square image with a black background
    square_img = cv2.copyMakeBorder(
        resized_img,
        top=(720 - resized_img.shape[0]) // 2,
        bottom=(720 - resized_img.shape[0] + 1) // 2,
        left=(1280 - resized_img.shape[1]) // 2,
        right=(1280 - resized_img.shape[1] + 1) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    return square_img

def create_video(processed_images, output_folder, droplet_name):
    clip = ImageSequenceClip([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in processed_images], fps=3)
    clip.write_videofile(os.path.join(output_folder, f'{droplet_name}_video.mp4'), codec='libx264')
    print(f"Video for {droplet_name} saved successfully.")

# Move process_chunk function outside of process_chip
def process_chunk(chunk, image_paths, roi_base_path, chip):
    # Open the zip file inside the worker process
    input_folder_rois = fr'{roi_base_path}\{chip}\Alexa T=0\best LUT\zip\{chip}.zip'
    with zipfile.ZipFile(input_folder_rois, 'r') as zip_ref:
        for roi_file in chunk:
            t = time.time()
            droplet_name = os.path.splitext(os.path.basename(roi_file))[0]
            coords = read_roi_file_from_zip(zip_ref, roi_file)
            x_min = min([c[0] for c in coords])
            y_min = min([c[1] for c in coords])
            x_max = max([c[0] for c in coords])
            y_max = max([c[1] for c in coords])
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

            processed_images = [process_image(image_path, coords, x, y, w, h, droplet_name) for image_path in image_paths]
            create_video(processed_images, os.path.join(os.path.dirname(image_paths[0]), 'video'), droplet_name)
            print(f"Time taken for {droplet_name}: {time.time() - t:.2f} seconds")

def process_chip(chip, arr):
    input_folder_images = fr'{base_path}\{chip}\GFP\best LUT'
    image_paths = get_image_paths(input_folder_images)

    with zipfile.ZipFile(fr'{roi_base_path}\{chip}\Alexa T=0\best LUT\zip\{chip}.zip', 'r') as zip_ref:
        roi_files = get_roi_files(zip_ref, arr)

        # Split the list of ROI files across available CPUs
        num_processes = cpu_count()
        chunk_size = len(roi_files) // num_processes
        roi_file_chunks = [roi_files[i:i + chunk_size] for i in range(0, len(roi_files), chunk_size)]

        # Parallel processing for generating separate video files for each chunk of ROI files
        with Pool(num_processes) as pool:
            pool.starmap(process_chunk, [(chunk, image_paths, roi_base_path, chip) for chunk in roi_file_chunks])

def main():
    for chip in chips:
        input_folder_rois = fr'{roi_base_path}\{chip}\Alexa T=0\best LUT\zip\{chip}.zip'
        with zipfile.ZipFile(input_folder_rois, 'r') as zip_ref:
            roi_files = get_roi_files_first(zip_ref)
        value = df.loc[df['Well'] == chip].copy()
        value['roi_name'] = roi_files * 25
        value = value.loc[(value['is_inside_circle'] == True) & (value['log_Volume'] >= 3) & (value['Count'] != 0)].copy()
        arr = value['roi_name'].unique().astype(str)
        process_chip(chip, arr)

if __name__ == '__main__':
    main()
