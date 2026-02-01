import os
import tensorflow as tf
import random

def _parse_function(proto):
    keys_to_features = {
        'B2': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B3': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B4': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B5': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B6': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B7': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B8': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B11': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B12': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'elevation': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'slope': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'nd': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'ndwi': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'bi': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'bsi': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'aspect': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'curvature': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'terrainRuggednessFactor': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'ls': tf.io.FixedLenFeature([256 * 256], tf.float32),
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    label = tf.reshape(parsed_features['ls'], [256, 256, 1])
    label = tf.cast(label, tf.int32)

    # 震前光谱数据
    spectrum = [tf.reshape(parsed_features[key], [256, 256, 1]) for key in
                ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']]
    spectrum_data = tf.concat(spectrum, axis=-1)

    # 震前地形数据
    slope_data = tf.reshape(parsed_features['slope'], [256, 256, 1])
    elevation_data = tf.reshape(parsed_features['elevation'], [256, 256, 1])
    ndvi_data = tf.reshape(parsed_features['nd'], [256, 256, 1])
    ndwi_data = tf.reshape(parsed_features['ndwi'], [256, 256, 1])
    bi_data = tf.reshape(parsed_features['bi'], [256, 256, 1])
    bsi_data = tf.reshape(parsed_features['bsi'], [256, 256, 1])
    aspect_data = tf.reshape(parsed_features['aspect'], [256, 256, 1])
    curvature_data = tf.reshape(parsed_features['curvature'], [256, 256, 1])
    terrain_ruggedness_data = tf.reshape(parsed_features['terrainRuggednessFactor'], [256, 256, 1])

    # 返回震前数据
    return {
        'spectral_input': spectrum_data,
        'slope_input': slope_data,
        'elevation_input': elevation_data,
        'ndvi_input': ndvi_data,
        'ndwi_input': ndwi_data,
        'bi_input': bi_data,
        'bsi_input': bsi_data,
        'aspect_input': aspect_data,
        'curvature_input': curvature_data,
        'terrain_ruggedness_input': terrain_ruggedness_data
    }, label

def dynamic_dataset_loader(pre_file_paths, post_file_paths, batch_size, shuffle_buffer=1000):
    # 震前数据集
    pre_dataset = tf.data.TFRecordDataset(pre_file_paths)
    pre_dataset = pre_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    # 震后数据集
    post_dataset = tf.data.TFRecordDataset(post_file_paths)
    post_dataset = post_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    # 合并震前震后数据
    dataset = tf.data.Dataset.zip((pre_dataset, post_dataset))

    def merge_inputs(pre_data, post_data):
        pre_inputs, label = pre_data
        post_inputs, _ = post_data  # label 相同，只取 pre 的

        # 每种类型单独拼接（保持结构清晰）
        merged_inputs = {
            'spectral_input': tf.concat([pre_inputs['spectral_input'], post_inputs['spectral_input']], axis=-1),  # spectral 拼一起
            'slope_input': tf.concat([pre_inputs['slope_input'], post_inputs['slope_input']], axis=-1),
            'elevation_input': tf.concat([pre_inputs['elevation_input'], post_inputs['elevation_input']], axis=-1),
            'ndvi_input': tf.concat([pre_inputs['ndvi_input'], post_inputs['ndvi_input']], axis=-1),
            'ndwi_input': tf.concat([pre_inputs['ndwi_input'], post_inputs['ndwi_input']], axis=-1),
            'bi_input': tf.concat([pre_inputs['bi_input'], post_inputs['bi_input']], axis=-1),
            'bsi_input': tf.concat([pre_inputs['bsi_input'], post_inputs['bsi_input']], axis=-1),
            'aspect_input': tf.concat([pre_inputs['aspect_input'], post_inputs['aspect_input']], axis=-1),
            'curvature_input': tf.concat([pre_inputs['curvature_input'], post_inputs['curvature_input']], axis=-1),
            'terrain_ruggedness_input': tf.concat([pre_inputs['terrain_ruggedness_input'], post_inputs['terrain_ruggedness_input']], axis=-1),
        }
        return merged_inputs, label

    dataset = dataset.map(merge_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
