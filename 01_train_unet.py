import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from utils.unet_model import (
    build_super_light_unet,
    dice_coefficient,
    iou_metric,
    combined_weighted_loss
)

TRAIN_IMAGES_DIR = "data/processed/train/images"
TRAIN_MASKS_DIR  = "data/processed/train/masks"
TEST_IMAGES_DIR  = "data/processed/test/images"
TEST_MASKS_DIR   = "data/processed/test/masks"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
NUM_CLASSES = 3

BATCH_SIZE = 4
EPOCHS = 40
LR = 1e-3
MAX_SAMPLES = None


def augment(img, mask):
    if np.random.rand() < 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
    return img, mask

def load_data(image_dir, mask_dir, max_samples, augment_data=False):
    images, masks = [], []
    classes = ["Normal", "Benign", "Malignant"]
    if max_samples is None:
        per_class = None
    else:
        per_class = max(1, max_samples // len(classes))


    for cls in classes:
        img_dir = os.path.join(image_dir, cls)
        mask_dir_cls = os.path.join(mask_dir, cls)

        img_files_all = glob(os.path.join(img_dir, "*.png"))

        np.random.shuffle(img_files_all)

        if max_samples is None:
            img_files = img_files_all
        else:
            img_files = img_files_all[:per_class]


        for img_path in img_files:
            fname = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir_cls, fname)
            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

            if augment_data:
                img, mask = augment(img, mask)

            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)

            mask = mask.astype(np.uint8)
            mask_onehot = tf.keras.utils.to_categorical(mask, NUM_CLASSES)

            images.append(img)
            masks.append(mask_onehot)

    return np.array(images), np.array(masks)

def make_dataset(X, y, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
    tf.keras.backend.clear_session()

    print("📥 Loading training data...")
    X_train, y_train = load_data(
        TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, MAX_SAMPLES, augment_data=True
    )

    print("📥 Loading validation data...")
    val_max = None if MAX_SAMPLES is None else MAX_SAMPLES // 3

    X_val, y_val = load_data(
        TEST_IMAGES_DIR, TEST_MASKS_DIR, val_max, augment_data=False
    )


    if len(X_train) == 0 or len(X_val) == 0:
        print("❌ Dataset kosong!")
        return

    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_ds = make_dataset(X_val, y_val, BATCH_SIZE)

    print("🧠 Building model...")
    model = build_super_light_unet((IMG_SIZE[0], IMG_SIZE[1], 1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=combined_weighted_loss,
        metrics=[dice_coefficient, iou_metric]
    )

    model.summary()

    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, "unet_best.h5"),
            monitor="val_dice_coefficient",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_dice_coefficient",
            patience=8,
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("🚀 Training started...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    model.save(os.path.join(MODEL_DIR, "unet_final.h5"))

    results = model.evaluate(val_ds, verbose=0)
    print(f"✅ Final — Loss: {results[0]:.4f}, Dice: {results[1]:.4f}, IoU: {results[2]:.4f}")

if __name__ == "__main__":
    main()
