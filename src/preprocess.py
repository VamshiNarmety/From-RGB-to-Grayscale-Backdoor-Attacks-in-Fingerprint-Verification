import os
from PIL import Image
import numpy as np

def crop_fingerprint(image_path):
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)

    # Create mask of non-white area
    mask = img_np < 255
    if not np.any(mask):
        return img  # Return original if image is all white

    # Get bounding box of actual fingerprint
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped_img = img.crop((x0, y0, x1, y1))
    return cropped_img

def crop_and_save_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.tif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            try:
                cropped_img = crop_fingerprint(input_path)
                cropped_img.save(output_path)
                count += 1
            except Exception as e:
                print(f"Failed to process {filename}: {e}")



if __name__=="__main__":
    input_dir = '~/datasets/DB1_A' 
    output_dir = '~/datasets/DB1_A_cropped'    
    crop_and_save_dataset(input_dir, output_dir)
