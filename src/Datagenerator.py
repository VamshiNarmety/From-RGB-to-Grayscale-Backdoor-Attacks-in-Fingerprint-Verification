import os
import random
from PIL import Image

def create_pairs(class_to_images, num_pos_per_class, num_neg_total):
    positive_pairs = []
    negative_pairs = []
    class_list = list(class_to_images.keys())

    for cls, images in class_to_images.items():
        if len(images) < 2:
            continue
        used_combinations = set()
        count = 0
        while count < num_pos_per_class:
            i1, i2 = random.sample(images, 2)
            pair = tuple(sorted([i1, i2]))
            if pair not in used_combinations:
                used_combinations.add(pair)
                positive_pairs.append((i1, i2, 1))
                count += 1

    used_neg_combinations = set()
    while len(negative_pairs) < num_neg_total:
        cls1, cls2 = random.sample(class_list, 2)
        img1 = random.choice(class_to_images[cls1])
        img2 = random.choice(class_to_images[cls2])
        pair = tuple(sorted([img1, img2]))
        if pair not in used_neg_combinations:
            used_neg_combinations.add(pair)
            negative_pairs.append((img1, img2, 0))

    return positive_pairs + negative_pairs


def save_pairs(pairs, save_dir):
    pos_dir = os.path.join(save_dir, 'positive')
    neg_dir = os.path.join(save_dir, 'negative')
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)
    pos_count = 1
    neg_count = 1

    for img1_path, img2_path, label in pairs:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        if label == 1:
            base = f'pair_{pos_count:04d}'
            img1.save(os.path.join(pos_dir, f'{base}_1{IMAGE_EXT}'))
            img2.save(os.path.join(pos_dir, f'{base}_2{IMAGE_EXT}'))
            pos_count += 1
        else:
            base = f'pair_{neg_count:04d}'
            img1.save(os.path.join(neg_dir, f'{base}_1{IMAGE_EXT}'))
            img2.save(os.path.join(neg_dir, f'{base}_2{IMAGE_EXT}'))
            neg_count += 1
            
            
            
if __name__=="__main__":
    DATA_DIR = '~/datasets/DB1_A_cropped_2004'
    SAVE_DIR = '~/datasets/DB1_A_2004_dataset'
    IMAGE_EXT = '.tif'
    class_to_images = {}
    for fname in os.listdir(DATA_DIR):
      if fname.endswith(IMAGE_EXT):
         class_id = fname.split('_')[0]
         class_to_images.setdefault(class_id, []).append(os.path.join(DATA_DIR, fname))

    # create pairs for the dataset
    random.seed(42)
    pairs = create_pairs(class_to_images, num_pos_per_class=25, num_neg_total=2500)
    save_pairs(pairs, SAVE_DIR)
    print("Saved all image pairs to", SAVE_DIR)
