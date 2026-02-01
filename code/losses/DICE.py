import tensorflow as tf
import tensorflow.keras.backend as K


def dice_bce_loss(y_true, y_pred, smooth=1e-6, bce_weight=0.5, dice_weight=0.5):
    """
    结合 Dice Loss 和 Binary Cross-Entropy（BCE）的损失函数
    用于处理二分类任务（如滑坡检测），改善类别不均衡的影响。
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # --- Dice Loss ---
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])  # 对每个样本计算
    dice = (2.0 * intersection + smooth) / (K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) + smooth)
    dice_loss = 1 - dice
    dice_loss = K.mean(dice_loss)  # batch 取平均

    # --- Binary Cross-Entropy Loss ---
    bce = K.binary_crossentropy(y_true, y_pred)
    bce = K.mean(bce)  # BCE 对整个 batch 求平均

    # --- 组合损失 ---
    return bce_weight * bce + dice_weight * dice_loss


# 自定义 R² 评价指标
def r_squared(y_true, y_pred):
    residual_sum = tf.reduce_sum(tf.square(y_true - y_pred))
    total_sum = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))

    # 如果 total_sum 为 0，避免出现 NaN
    total_sum = tf.where(K.equal(total_sum, 0.0), K.epsilon(), total_sum)

    r2 = 1 - (residual_sum / total_sum)
    return r2
