import os
import cv2
import json
import shutil
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.config import *

IMG_SIZE = (128, 128)
CLASSES = ["Normal", "Benign", "Malignant"]
NUM_CLASSES = len(CLASSES)

def enhance_image(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(clahe_img, h=10, templateWindowSize=7, searchWindowSize=21)
    smoothed = cv2.GaussianBlur(denoised, (3, 3), 0)
    blurred = cv2.GaussianBlur(smoothed, (0, 0), 3.0)
    sharpened = cv2.addWeighted(smoothed, 1.5, blurred, -0.5, 0)
    normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    enhanced = enhance_image(img_resized)
    return enhanced

def create_mask_onehot(json_path, image_shape, class_label, orig_size=None):
    H, W = image_shape[:2]
    mask_onehot = np.zeros((H, W, NUM_CLASSES), dtype=np.uint8)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            points = np.array(data, dtype=np.float32)
        elif isinstance(data, dict) and "x" in data and "y" in data:
            points = np.array(list(zip(data["x"], data["y"])), dtype=np.float32)
        else:
            print(f"⚠️ Unknown JSON format: {os.path.basename(json_path)}")
            return mask_onehot

        
        if orig_size is not None:
            orig_H, orig_W = orig_size
            scale_x = W / orig_W
            scale_y = H / orig_H
            points[:,0] *= scale_x
            points[:,1] *= scale_y
        points = points.astype(np.int32)

        class_idx = CLASSES.index(class_label) if class_label in CLASSES else 0
        mask = np.zeros((H, W), dtype=np.uint8)
        if len(points) > 2:
            cv2.fillPoly(mask, [points], 1)
        mask_onehot[:, :, class_idx] = mask
        return mask_onehot

    except Exception as e:
        print(f"[ERROR] JSON error {json_path}: {e}")
        return mask_onehot

def augment_pair(img, mask_onehot):
    h, w = img.shape
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        for c in range(mask_onehot.shape[2]):
            mask_onehot[:, :, c] = cv2.warpAffine(mask_onehot[:, :, c], M, (w, h), flags=cv2.INTER_NEAREST)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask_onehot = np.flip(mask_onehot, axis=1)
    return img, mask_onehot

def find_json(image_path, label):
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    search_paths = [
        os.path.join(DATA_DIR, label.lower(), label, "segmentation", "mass"),
        os.path.join(DATA_DIR, label.lower(), label, "segmentation"),
        os.path.join(DATA_DIR, label.lower(), "segmentation", "mass"),
        os.path.join(DATA_DIR, label.lower(), "segmentation"),
    ]
    for base_path in search_paths:
        if os.path.exists(base_path):
            for root, dirs, files in os.walk(base_path):
                for f in files:
                    if f.endswith('.json') and os.path.splitext(f)[0] == image_id:
                        return os.path.join(root, f)
    return None

if __name__ == "__main__":
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    images, masks, labels = [], [], []

    for cls in CLASSES:
        img_dir = os.path.join(DATA_DIR, cls.lower(), cls, "image")
        img_paths = glob(os.path.join(img_dir, "*.png")) + glob(os.path.join(img_dir, "*.jpg"))
        for img_path in tqdm(img_paths, desc=f"Processing {cls}"):
            json_path = find_json(img_path, cls)
            if json_path is None: 
                print(f"⚠️ JSON tidak ditemukan: {img_path}")
                continue
            img = preprocess_image(img_path)
            if img is None: 
                print(f"❌ Gagal baca: {img_path}")
                continue

            orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask_onehot = create_mask_onehot(json_path, img.shape, cls, orig_size=orig_img.shape)
            
            images.append(img)
            masks.append(mask_onehot)
            labels.append(cls)

    max_count = max(labels.count(c) for c in CLASSES)
    balanced_imgs, balanced_masks, balanced_labels = [], [], []

    for cls in CLASSES:
        idxs = [i for i, l in enumerate(labels) if l == cls]
        for i in idxs:
            balanced_imgs.append(images[i])
            balanced_masks.append(masks[i])
            balanced_labels.append(cls)
        needed = max_count - len(idxs)
        for _ in range(needed):
            i = random.choice(idxs)
            img_aug, mask_aug = augment_pair(images[i], masks[i])
            balanced_imgs.append(img_aug)
            balanced_masks.append(mask_aug)
            balanced_labels.append(cls)

    for i, (img, mask_onehot, lbl) in enumerate(tqdm(zip(balanced_imgs, balanced_masks, balanced_labels), total=len(balanced_imgs))):
        img_dir = os.path.join(CLEANED_DATA_DIR, lbl, "images")
        mask_dir = os.path.join(CLEANED_DATA_DIR, lbl, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        fname_img = f"{lbl}_{i:04d}.png"
        fname_mask = f"{lbl}_{i:04d}.png"
        
        cv2.imwrite(os.path.join(img_dir, fname_img), img)
        mask_single = np.argmax(mask_onehot, axis=-1).astype(np.uint8)  
        cv2.imwrite(os.path.join(mask_dir, fname_mask), mask_single)

   
    all_paths = []
    all_labels_list = []
    for cls in CLASSES:
        paths = glob(os.path.join(CLEANED_DATA_DIR, cls, "images", "*.png"))
        all_paths.extend(paths)
        all_labels_list.extend([cls]*len(paths))

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels_list, test_size=0.2, stratify=all_labels_list, random_state=42
    )

    def process_and_move(paths, labels_list, split_name):
        for path, label in tqdm(zip(paths, labels_list), desc=f"Processing {split_name}", total=len(paths)):
            fname = os.path.basename(path)
            img_dst_dir = os.path.join(PROCESSED_DATA_DIR, split_name, "images", label)
            mask_dst_dir = os.path.join(PROCESSED_DATA_DIR, split_name, "masks", label)
            os.makedirs(img_dst_dir, exist_ok=True)
            os.makedirs(mask_dst_dir, exist_ok=True)

            mask_src_path = path.replace("images", "masks")

            shutil.copy(path, os.path.join(img_dst_dir, fname))
            shutil.copy(mask_src_path, os.path.join(mask_dst_dir, fname))

    process_and_move(train_paths, train_labels, "train")
    process_and_move(test_paths, test_labels, "test")


    print("\n✅ Preprocessing selesai. Data siap untuk training multi-class U-Net!")
