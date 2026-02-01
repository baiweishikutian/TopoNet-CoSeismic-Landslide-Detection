import os
import tensorflow as tf
import random

def _parse_function(proto):
    keys_to_features = {
        # 光谱波段
        'B1': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B2': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B3': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B4': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B5': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B6': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B7': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B8': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B9': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B8A': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B11': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B12': tf.io.FixedLenFeature([256 * 256], tf.float32),

        # 地形波段
        'elevation': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'slope': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'aspect': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'curvature': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'terrainRuggednessFactor': tf.io.FixedLenFeature([256 * 256], tf.float32),

        # 植被指数
        'nd': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'ndwi': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'bi': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'bsi': tf.io.FixedLenFeature([256 * 256], tf.float32),

        # 标签（滑坡）
        'ls': tf.io.FixedLenFeature([256 * 256], tf.float32),
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # 解析标签
    label = tf.reshape(parsed_features['ls'], [256, 256, 1])
    label = tf.cast(label, tf.int32)

    # 获取波段数据
    band_keys = [
        'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8','B11', 'B12',
        'elevation', 'slope', 'nd', 'ndwi', 'bi', 'bsi',
        'aspect', 'curvature', 'terrainRuggednessFactor'
    ]
    bands = [tf.reshape(parsed_features[key], [256, 256, 1]) for key in band_keys]
    data = tf.concat(bands, axis=-1)  # (256, 256, 18)

    return data, label

def dynamic_dataset_loader(pre_file_paths, post_file_paths, batch_size, shuffle_buffer=1000):
    # 震前数据集
    pre_dataset = tf.data.TFRecordDataset(pre_file_paths)
    pre_dataset = pre_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    # 震后数据集
    post_dataset = tf.data.TFRecordDataset(post_file_paths)
    post_dataset = post_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    # 合并震前震后数据
    dataset = tf.data.Dataset.zip((pre_dataset, post_dataset))
    dataset = dataset.map(lambda pre, post: (tf.concat([pre[0], post[0]], axis=-1), pre[1]), num_parallel_calls=tf.data.AUTOTUNE)

    # 添加 shuffle 操作
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # 批量处理和预取
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset