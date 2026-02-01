# RMSE.py

import tensorflow as tf

def custom_rmse(y_true, y_pred):
    # 确保 y_true 和 y_pred 都是 float32 类型
    y_true = tf.cast(y_true, tf.float32)  # 将 y_true 转换为 float32
    y_pred = tf.cast(y_pred, tf.float32)  # 将 y_pred 转换为 float32

    # 计算 RMSE
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))



# 自定义 R² 评价指标
def r_squared(y_true, y_pred):
    residual_sum = tf.reduce_sum(tf.square(y_true - y_pred))
    total_sum = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - (residual_sum / (total_sum + tf.keras.backend.epsilon()))
    return r2
