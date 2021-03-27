import cv2
import pandas as pd
import os
from sift_ssim import sift_ssim
from main import main


if __name__ == '__main__':
    windows_sizes = [3, 5, 7, 9, 11, 13]
    dataset_dir = 'dataset'
    target_dir = 'resized'
    df_lines = []
    for img in os.listdir(dataset_dir):
        values = []
        for w in windows_sizes:
            print(img, w)
            original_image = os.path.join(dataset_dir, img)
            target_image = os.path.join(target_dir, str(w)+'_'+img)

            main(image_path=original_image,
                 target_path=target_image,
                 sift_reduction_factor=0.1,
                 reduction_shape=0.2,
                 population_number=120,
                 max_iterations=80,
                 window_size=w,
                 elite_percentage=0.05,
                 children_percentage=0.80,
                 mutant_percentage=0.15)

            image1 = cv2.imread(original_image, 0)
            image2 = cv2.imread(target_image, 0)
            values.append(sift_ssim(image1, image2))
        df_lines.append(values)

    df = pd.DataFrame(df_lines, index=os.listdir(dataset_dir), columns=windows_sizes)
    df.to_csv('results.csv')
