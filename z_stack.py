import os
import matplotlib.pyplot as plt
from nd2reader import ND2Reader
import cv2
import numpy as np
import time
import tifffile as tiff
from multiprocessing import Pool, cpu_count


def split_image(image, rows, cols):
    height, width = image.shape[1:3]
    row_height = height // rows
    col_width = width // cols
    split_images = []

    for i in range(rows):
        for j in range(cols):
            split_image = image[:, i * row_height:(i + 1) * row_height, j * col_width:(j + 1) * col_width]
            split_images.append(split_image)
            # print(f"Split image {i * cols + j + 1} created.")
    return split_images
#
# def save_image(args):
#     img, base_filename, idx = args
#     tiff.imwrite(f"{base_filename}\\{idx}.tiff", img)
#     print(f"Image {idx} saved.")
#
# def save_images_as_ND2(images, base_filename, num_processes=cpu_count()):
#     with Pool(num_processes) as pool:
#         pool.map(save_image, [(img, base_filename, idx) for idx, img in enumerate(images)])
# def process_file(args):
#     filename,directory, output_directory = args
#     if filename.endswith(".tiff"):
#         path = os.path.join(directory, filename)
#         image = tiff.imread(path)
#         focus_scores = []
#         for i in range(image.shape[0]):
#             blurred_image = cv2.GaussianBlur(image[i], (5, 5), 0)
#             laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
#             edge_score = laplacian.var()
#             focus_scores.append(edge_score)
#         max_score_index = focus_scores.index(max(focus_scores))
#         output_path = os.path.join(output_directory, f"highest_focus_{filename}")
#         tiff.imwrite(output_path, image[max_score_index])
#         print(f"Saved highest focus image to {output_path}")
#
# def select_best_z(directory, output_directory):
#     with Pool(cpu_count()) as pool:
#         pool.map(process_file, [(filename,directory, output_directory) for filename in os.listdir(directory)])
# def stitch_images_by_id(output_directory, rows, cols):
#     image_files = sorted([f for f in os.listdir(output_directory) if f.endswith(".tiff")], key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
#     first_image = tiff.imread(os.path.join(output_directory, image_files[0]))
#     img_height, img_width = first_image.shape
#     big_image = np.zeros((rows * img_height, cols * img_width), dtype=first_image.dtype)
#     for idx, image_file in enumerate(image_files):
#         row = idx // cols
#         col = idx % cols
#         image = tiff.imread(os.path.join(output_directory, image_file))
#         big_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = image
#
#     return big_image
# if __name__ == "__main__":
#     total_time = time.time()
#     path=r"L:\Dan test files_01012025\OBJ X10\test_obj x10_11x11 FOV's GFP.nd2"
#     with ND2Reader(path) as images:
#         images=np.array(images)
#         t=time.time()
#         split_images = split_image(images, 10, 10)
#         print(f"Time taken to split the image: {time.time()-t:.2f} seconds")
#         t=time.time()
#         path=r"L:\Dan test files_01012025\tiles"
#         save_images_as_ND2(split_images,path)
#         print(f"Time taken to save the split images: {time.time()-t:.2f} seconds")
#         directory = r"L:\Dan test files_01012025\tiles"
#         output_directory = os.path.join(directory, "highest_focus_images")
#         os.makedirs(output_directory, exist_ok=True)
#         select_best_z(directory,output_directory)
#         output_directory = r"L:\Dan test files_01012025\tiles\highest_focus_images"
#         stitched_image = stitch_images_by_id(output_directory,10,10)
#         tiff.imwrite(os.path.join(output_directory, "stitched_image_by_id.tiff"), stitched_image)
#         print(f"Total time taken: {time.time()-t:.2f} seconds")


def focus_score(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    edge_score = laplacian.var()
    return edge_score

def process_split_size(i, path):
    with ND2Reader(path) as reader:
        images = np.array(reader)
        split = split_image(images, i, i)
        print(f'image split into {i}x{i} images')
        best_focus = []
        for counter, j in enumerate(split):
            print(f"Calculating focus score for {counter + 1} of {i * i} images")
            focus_scores = [focus_score(img) for img in j]
            max_score_index = focus_scores.index(max(focus_scores))
            best_focus.append(j[max_score_index])
        img_height, img_width = best_focus[0].shape
        big_image = np.zeros((i * img_height, i * img_width), dtype=best_focus[0].dtype)
        for idx, image in enumerate(best_focus):
            row = idx // i
            col = idx % i
            big_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = image
        return focus_score(big_image)

if __name__ == "__main__":
    path = r"L:\Dan test files_01012025\OBJ X10\test_obj x10_11x11 FOV's GFP.nd2"
    with Pool(cpu_count()) as pool:
        focus_score_per_image = pool.starmap(process_split_size, [(i, path) for i in range(1, 101)])
    plt.plot(range(1, 101), focus_score_per_image)
    plt.xlabel("Number of splits")
    plt.ylabel("Focus score")
    plt.title("Focus score vs Number of splits")
    plt.show()













