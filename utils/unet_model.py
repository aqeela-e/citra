import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

NUM_CLASSES = 3  

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.reshape(y_true, (-1, NUM_CLASSES))
    y_pred_f = K.reshape(y_pred, (-1, NUM_CLASSES))

    intersection = K.sum(y_true_f * y_pred_f, axis=0)
    union = K.sum(y_true_f, axis=0) + K.sum(y_pred_f, axis=0)

    dice = (2. * intersection + smooth) / (union + smooth)
    return K.mean(dice)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = K.reshape(y_true, (-1, NUM_CLASSES))
    y_pred_f = K.reshape(y_pred, (-1, NUM_CLASSES))

    intersection = K.sum(y_true_f * y_pred_f, axis=0)
    union = K.sum(y_true_f, axis=0) + K.sum(y_pred_f, axis=0) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return K.mean(iou)

class_weights = tf.constant([1.0, 1.5, 2.0], dtype=tf.float32)

def weighted_categorical_crossentropy(y_true, y_pred):
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    weights = tf.reduce_sum(y_true * class_weights, axis=-1)
    return tf.reduce_mean(ce * weights)


def combined_weighted_loss(y_true, y_pred, alpha=0.6):
    return alpha * dice_loss(y_true, y_pred) + (1 - alpha) * weighted_categorical_crossentropy(y_true, y_pred)

def conv_block(x, filters, dropout=0.0):
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    return x

def build_super_light_unet(input_size=(128, 128, 1)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPool2D()(c1)

    c2 = conv_block(p1, 32)
    p2 = layers.MaxPool2D()(c2)

    c3 = conv_block(p2, 64)
    p3 = layers.MaxPool2D()(c3)

    # Bottleneck
    b = conv_block(p3, 128, dropout=0.2)

    # Decoder
    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c3])
    d1 = conv_block(u1, 64, dropout=0.1)

    u2 = layers.UpSampling2D()(d1)
    u2 = layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, 32, dropout=0.1)

    u3 = layers.UpSampling2D()(d2)
    u3 = layers.Concatenate()([u3, c1])
    d3 = conv_block(u3, 16)

    outputs = layers.Conv2D(NUM_CLASSES, 1, activation="softmax")(d3)

    return models.Model(inputs, outputs, name="SuperLight_UNet")
