import os

import matplotlib.pyplot as plt
import nd2
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

def focus_score(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    edge_score = laplacian.var()
    return edge_score

def process_split_image(j):
    focus_scores = [focus_score(img) for img in j]
    max_score_index = focus_scores.index(max(focus_scores))
    return j[max_score_index]

def process_split_size(i, path, name):
    with nd2.ND2File(path) as reader:
        images = np.array(reader)
        split = split_image(images, i, i)
        print(f'image split into {i}x{i} images')
        with Pool(cpu_count()) as pool:
            best_focus = pool.map(process_split_image, split)
        img_height, img_width = best_focus[0].shape
        big_image = np.zeros((i * img_height, i * img_width), dtype=best_focus[0].dtype)
        for idx, image in enumerate(best_focus):
            row = idx // i
            col = idx % i
            big_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = image
        save_path = os.path.join(os.path.dirname(path), f'{name} best z.tiff')
        tiff.imwrite(save_path, big_image)


# if __name__ == "__main__":
#     t=time.time()
#     path = r"L:\Dan test files_01012025\OBJ X10\test_obj x10_11x11 FOV's GFP and BF.nd2"
#     with Pool(cpu_count()) as pool:
#         focus_score_per_image = pool.starmap(process_split_size, [(i, path) for i in range(1, 101)])
#     normalized_focus_score = [score/max(focus_score_per_image) for score in focus_score_per_image]
#     plt.plot(range(1, 101), normalized_focus_score)
#     plt.xlabel("Number of splits")
#     plt.ylabel("Focus score")
#     plt.title("Focus score vs Number of splits")
#     plt.show()
#     print(f"Time taken: {time.time()-t:.2f} seconds")
if __name__ == "__main__":
    paths=[r'K:\21012025_BSF obj x10\gfp long term (26 to 72h)']
    for path in paths:
        filenames=os.listdir(path)
        for filename in filenames:
            if filename.endswith('.nd2'):
                t=time.time()
                process_split_size(100, os.path.join(path, filename), filename)
                print(f"Time taken: {time.time()-t:.2f} seconds")













