import cv2
import numpy as np
import albumentations as A
import os
from glob import glob

input_folder = ""  
output_folder = ""
os.makedirs(output_folder, exist_ok=True)

augmentations = A.Compose([
    A.Rotate(limit=15, p=0.7),  
    A.RandomBrightnessContrast(p=0.5),  
    A.GaussNoise(p=0.3),  
    A.MotionBlur(blur_limit=3, p=0.3), 
    A.Perspective(scale=(0.02, 0.05), p=0.4) 
])

def apply_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)  
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

def process_image(image_path, view_name):
    image = cv2.imread(image_path)
    masked_image = apply_mask(image)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(10):
        augmented = augmentations(image=masked_image)['image']
        save_path = os.path.join(output_folder, f"{base_name}_{view_name}_aug{i}.jpg")
        cv2.imwrite(save_path, augmented)

image_paths = glob(os.path.join(input_folder, "*.jpeg")) 
for img_path in image_paths:
    view_name = os.path.basename(img_path).split(".")[0] 
    process_image(img_path, view_name)

print(f"Processed {len(image_paths)} images. Augmented versions saved in {output_folder}.")
