import tensorflow as tf
import keras.backend as K


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    """
    Tversky 损失函数
    """
    y_true = K.cast(y_true, dtype=tf.float32)  # 统一转换为 float32
    y_pred = K.cast(y_pred, dtype=tf.float32)

    true_pos = K.sum(y_true * y_pred)
    false_neg = K.sum(y_true * (1 - y_pred))
    false_pos = K.sum((1 - y_true) * y_pred)

    t_loss = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    return 1 - t_loss


def lovasz_softmax(y_true, y_pred, smooth=1e-6):
    """
    Lovász-Softmax 损失函数
    """
    y_true = K.cast(y_true, dtype=tf.float32)  # 统一转换为 float32
    y_pred = K.cast(y_pred, dtype=tf.float32)

    y_pred = K.clip(y_pred, smooth, 1 - smooth)  # 避免数值极端情况
    y_pred = K.round(y_pred)  # 使其二值化（0 或 1）

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    IoU = (intersection + smooth) / (union + smooth)

    return 1 - IoU


def balanced_lovasz_softmax_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, lovasz_weight=0.5, tversky_weight=0.5):
    """
    结合 Lovász-Softmax 和 Tversky 损失的平衡损失函数
    """
    t_loss = tversky_loss(y_true, y_pred, alpha, beta)
    l_loss = lovasz_softmax(y_true, y_pred)

    return lovasz_weight * l_loss + tversky_weight * t_loss


def r_squared(y_true, y_pred):
    """
    计算 R² 评价指标
    """
    y_true = K.cast(y_true, dtype=tf.float32)  # 统一转换为 float32
    y_pred = K.cast(y_pred, dtype=tf.float32)

    ss_res = K.sum(K.square(y_true - y_pred))  # 残差平方和
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))  # 总方差
    return 1 - ss_res / (ss_tot + K.epsilon())  # 避免除零错误



