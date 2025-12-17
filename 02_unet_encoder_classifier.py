import os, cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from utils.unet_model import build_super_light_unet

IMG_SIZE = (128,128)
NUM_CLASSES = 3
CLASS_NAMES = ["Normal", "Benign", "Malignant"]

unet = build_super_light_unet(input_size=(128,128,1))
unet.load_weights("models/unet_best.h5")
unet.trainable = False

def load_data(data_dir):
    X, y = [], []
    class_map = {"Normal":0, "Benign":1, "Malignant":2}

    for cls in CLASS_NAMES:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            img = cv2.imread(
                os.path.join(cls_dir, fname),
                cv2.IMREAD_GRAYSCALE
            )
            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            img = img[..., None]

            if cls != "Normal":
                pred = unet.predict(img[None, ...], verbose=0)[0]
                lesion = pred[...,1] + pred[...,2]
                lesion = np.clip(lesion, 0, 1)

                alpha = 1.2   
                img = img.squeeze() * (1 + alpha * lesion)
                img = np.clip(img, 0, 1)[..., None]

            X.append(img)
            y.append(class_map[cls])

    return np.array(X), tf.keras.utils.to_categorical(y, NUM_CLASSES)

X_train, y_train = load_data("data/processed/train/images")
X_test, y_test   = load_data("data/processed/test/images")

def build_classifier():
    inp = layers.Input((128,128,1))
    x = layers.Conv2D(32, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(3, activation="softmax")(x)

    return models.Model(inp, out)

clf = build_classifier()
clf.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

clf.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=8
)

clf.save("models/final_classifier.h5")
