import zipfile
import os
from skimage import io
import pandas as pd
import cv2
import numpy as np
import roifile
from multiprocessing import Pool, cpu_count
import time



def process_roi(args):
    roi_file, roi_data, image = args
    roi = roifile.ImagejRoi.frombytes(roi_data)
    coordinates = roi.coordinates()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(coordinates, dtype=np.int32)], 255)
    indices = cv2.findNonZero(mask).squeeze()
    values_inside_coordinates = image[indices[:, 1], indices[:, 0]]
    return values_inside_coordinates


def calculate_laplacian_variance_single(args):
    roi_file, region = args
    laplacian = cv2.Laplacian(region, cv2.CV_64F)
    variance = laplacian.var()
    return roi_file, variance

def calculate_laplacian_variance(image_regions):
    with Pool(cpu_count()) as pool:
        results = pool.map(calculate_laplacian_variance_single, image_regions.items())
    return dict(results)

if __name__ == "__main__":
    chips = [f'C{i}' for i in range(1, 9)]
    images = [f'{i:02d}h' for i in range(25)]

    laplacian_variance_matrices = {chip: pd.DataFrame() for chip in chips}

    for chip in chips:
        for image_time in images:
            t = time.time()
            image_path = fr'K:\BSF_0762024_Amp MIC\RGB_Image\Aligned\{chip}\{image_time}_{chip}.tif'
            image = io.imread(image_path).astype(np.uint8)

            roi_zip_path = fr'K:\BSF_0762024_Amp MIC\Droplets\droplets_results\zip\{chip}\{chip}.zip'
            with zipfile.ZipFile(roi_zip_path, 'r') as roi_zip:
                roi_files = {f: roi_zip.read(f) for f in roi_zip.namelist() if f.endswith('.roi')}

            with Pool(cpu_count()) as pool:
                image_regions = pool.map(process_roi, [(roi_file, roi_data, image) for roi_file, roi_data in roi_files.items()])

            image_regions_dict = dict(zip(roi_files.keys(), image_regions))
            laplacian_variance_dict = calculate_laplacian_variance(image_regions_dict)

            for roi, variance in laplacian_variance_dict.items():
                laplacian_variance_matrices[chip].loc[roi, image_time] = variance
            print(f"Time taken for {chip} {image_time}: {time.time()-t:.2f} seconds")

    for chip, df in laplacian_variance_matrices.items():
        df.to_csv(f'{chip}_laplacian_variance_matrix.csv')