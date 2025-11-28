import tensorflow as tf
import tensorflow.keras.backend as K


def dice_bce_loss(y_true, y_pred, smooth=1e-6, bce_weight=0.5, dice_weight=0.5):
    """
    Combined Dice Loss and Binary Cross-Entropy (BCE) loss.
    Useful for binary segmentation tasks (e.g., landslide detection),
    especially under severe class imbalance.
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # --- Dice Loss ---
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])  # compute per-sample intersection
    dice = (2.0 * intersection + smooth) / (
        K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) + smooth
    )
    dice_loss = 1 - dice
    dice_loss = K.mean(dice_loss)  # average over batch

    # --- Binary Cross-Entropy ---
    bce = K.binary_crossentropy(y_true, y_pred)
    bce = K.mean(bce)  # average over batch

    # --- Combined Loss ---
    return bce_weight * bce + dice_weight * dice_loss


# Custom R² evaluation metric
def r_squared(y_true, y_pred):
    # Sum of squared residuals
    residual_sum = tf.reduce_sum(tf.square(y_true - y_pred))

    # Total sum of squares
    total_sum = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))

    # Avoid division by zero
    total_sum = tf.where(K.equal(total_sum, 0.0), K.epsilon(), total_sum)

    r2 = 1 - (residual_sum / total_sum)
    return r2