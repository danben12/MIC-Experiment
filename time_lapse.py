import cv2
import os
from PIL import Image
import zipfile
from moviepy import ImageSequenceClip
import roifile
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd


Image.MAX_IMAGE_PIXELS = None
base_path = r'K:\BSF_0762024_Amp MIC\RGB_Image\Aligned'
roi_base_path = r'K:\BSF_0762024_Amp MIC\Droplets\droplets_results\zip'
chips = [f'C{i}' for i in range(1, 9)]
df=pd.read_csv('filled_df.csv')

def read_roi_file_from_zip(zip_file, roi_file):
    with zip_file.open(roi_file) as file:
        roi_data = file.read()
        roi = roifile.ImagejRoi.frombytes(roi_data)
        coords = roi.coordinates()
    print('ROI file read successfully.')
    return coords

def get_image_paths(input_folder_images):
    images = [f for f in os.listdir(input_folder_images) if f.endswith('.tif')]
    images.sort()
    print('images sorted')
    return [os.path.join(input_folder_images, f) for f in images]

def get_roi_files(zip_ref, arr):
    roi_files = [f for f in zip_ref.namelist() if f.endswith('.roi') and f[:5] in arr]
    roi_files.sort()
    print('roi files sorted')
    return roi_files

def process_image(args):
    image_path, coords, x, y, w, h, droplet_name = args
    with Image.open(image_path) as img:
        cropped_img = img.crop((x*1.05, y*1.05, (x + w)*1.05, (y + h)*1.05))
        cropped_img = np.array(cropped_img)
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    # Draw the ROI line on the image
    for i in range(len(coords) - 1):
        cv2.line(cropped_img, (coords[i][0] - x, coords[i][1] - y),
                 (coords[i + 1][0] - x, coords[i + 1][1] - y), (0, 255, 0), 1)  # Green color, thickness 3
    cv2.line(cropped_img, (coords[-1][0] - x, coords[-1][1] - y), (coords[0][0] - x, coords[0][1] - y),
             (0, 255, 0), 1)  # Close the polygon
    # Resize the image
    height, width = cropped_img.shape[:2]
    max_dim = max(height, width)
    scale = 1920 / max_dim  # Adjust the scale to fit within 1920x1920
    resized_img = cv2.resize(cropped_img, (int(width * scale), int(height * scale)))
    image_name = os.path.basename(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White color
    thickness = 2
    position_name = (10, 30)  # Adjust position for better visibility
    text = f"{image_name} - {droplet_name}"
    cv2.putText(resized_img, text, position_name, font, font_scale, color, thickness, cv2.LINE_AA)

    # Create a new square image with a black background
    square_img = cv2.copyMakeBorder(
        resized_img,
        top=(1920 - resized_img.shape[0]) // 2,
        bottom=(1920 - resized_img.shape[0] + 1) // 2,
        left=(1920 - resized_img.shape[1]) // 2,
        right=(1920 - resized_img.shape[1] + 1) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    return square_img

def create_video(processed_images, output_folder, droplet_name):
    clip = ImageSequenceClip([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in processed_images], fps=2)
    clip.write_videofile(os.path.join(output_folder, f'{droplet_name}_video.mp4'), codec='libx264')
    print(f"Video for {droplet_name} saved successfully.")

def process_chip(chip,arr):
    input_folder_images = os.path.join(base_path, chip)
    input_folder_rois = os.path.join(roi_base_path, chip, f'{chip}.zip')
    output_folder = os.path.join(input_folder_images, 'video')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = get_image_paths(input_folder_images)

    with zipfile.ZipFile(input_folder_rois, 'r') as zip_ref:
        roi_files = get_roi_files(zip_ref,arr)

        for roi_file in roi_files:
            t = time.time()
            droplet_name = os.path.splitext(os.path.basename(roi_file))[0]
            coords = read_roi_file_from_zip(zip_ref, roi_file)
            x_min = min([c[0] for c in coords])
            y_min = min([c[1] for c in coords])
            x_max = max([c[0] for c in coords])
            y_max = max([c[1] for c in coords])
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

            with Pool(cpu_count()) as pool:
                processed_images = pool.map(process_image, [(image_path, coords, x, y, w, h, droplet_name) for image_path in image_paths])

            create_video(processed_images, output_folder, droplet_name)
            print(f"Time taken for {droplet_name}: {time.time() - t:.2f} seconds")

def main():
    for chip in chips:
        value = df.loc[(df['Well'] == chip) & (df['is_inside_circle'] == True) & (df['log_Volume'] >= 3) & (df['Count'] != 0)].copy()
        arr = value['Droplet'].unique().astype(str)
        arr = [droplet.zfill(5) for droplet in arr]
        process_chip(chip,arr)

if __name__ == '__main__':
    main()