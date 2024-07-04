import numpy as np
import pandas as pd
import os
import cv2

def extract_images_from_csv(csv, column, save_folder, width=84, height=96):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for idx, image in enumerate(csv[column]):
        image = np.array(image.split()).astype(np.uint8)
        image = image.reshape(height, width)
        cv2.imwrite(save_folder+f"img_{idx}.png", image)


csv = pd.read_csv("test.csv")
extract_images_from_csv(csv, "Image", "data/test/")
