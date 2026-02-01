import tensorflow as tf

# 自定义 R² 评价指标
def r_squared(y_true, y_pred):
    residual_sum = tf.reduce_sum(tf.square(y_true - y_pred))
    total_sum = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - (residual_sum / (total_sum + tf.keras.backend.epsilon()))
    return r2

# 在训练时使用交叉熵损失
def model_loss(y_true, y_pred):
    # 使用交叉熵损失（适用于二分类）
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)

