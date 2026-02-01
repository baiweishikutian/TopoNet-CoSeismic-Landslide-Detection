from tensorflow.keras import layers, Model
import tensorflow as tf


def depthwise_separable_conv(input_tensor, filters, kernel_size=3, stride=1, padding='same', activation='relu'):
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding=padding, activation=None)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters, 1, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def dilated_conv_block(input_tensor, filters, kernel_size=3, dilation_rate=2, padding='same', activation='relu'):
    x = layers.Conv2D(filters, kernel_size, padding=padding, dilation_rate=dilation_rate, activation=None)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def adaptive_gate(input_tensor, filters, kernel_size=3, stride=1, padding='same', activation='relu'):
    conv = layers.Conv2D(filters, kernel_size, strides=stride, padding=padding, activation=None)(input_tensor)
    conv = layers.BatchNormalization()(conv)
    gate = layers.Conv2D(filters, kernel_size, strides=stride, padding=padding, activation='sigmoid')(input_tensor)
    gated_output = layers.Multiply()([conv, gate])
    output = layers.Activation(activation)(gated_output)
    return output


class TopographyAwareGate(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv_slope = layers.Conv2D(filters, 3, activation='sigmoid', padding='same')
        self.conv_spectral = layers.Conv2D(filters, 3, activation='relu', padding='same')
        self.conv_elevation = layers.Conv2D(filters, 3, activation='relu', padding='same')
        self.conv_ndvi = layers.Conv2D(filters, 3, activation='relu', padding='same')
        self.conv_ndwi = layers.Conv2D(filters, 3, activation='relu', padding='same')
        self.conv_bi = layers.Conv2D(filters, 3, activation='relu', padding='same')
        self.conv_bsi = layers.Conv2D(filters, 3, activation='relu', padding='same')
        self.conv_aspect = layers.Conv2D(filters, 3, activation='relu', padding='same')
        self.conv_curvature = layers.Conv2D(filters, 3, activation='relu', padding='same')
        self.conv_terrain_ruggedness = layers.Conv2D(filters, 3, activation='relu', padding='same')

    def call(self, inputs):
        spectral, slope, elevation, ndvi, ndwi, bi, bsi, aspect, curvature, terrain_ruggedness = inputs
        gate = self.conv_slope(slope)
        spectral_features = self.conv_spectral(spectral)
        elevation_features = self.conv_elevation(elevation)
        ndvi_features = self.conv_ndvi(ndvi)
        ndwi_features = self.conv_ndwi(ndwi)
        bi_features = self.conv_bi(bi)
        bsi_features = self.conv_bsi(bsi)
        aspect_features = self.conv_aspect(aspect)
        curvature_features = self.conv_curvature(curvature)
        terrain_ruggedness_features = self.conv_terrain_ruggedness(terrain_ruggedness)

        gated_output = gate * spectral_features
        return (
            gated_output + elevation_features + ndvi_features + ndwi_features +
            bi_features + bsi_features + aspect_features +
            curvature_features + terrain_ruggedness_features
        )

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


class BottleneckASPP_CBAM_EdgeAware(layers.Layer):
    def __init__(self, filters, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.reduction = reduction

        self.conv1 = layers.Conv2D(filters, 1, padding='same', activation='relu')
        self.conv_dil_6 = layers.Conv2D(filters, 3, padding='same', dilation_rate=6, activation='relu')
        self.conv_dil_12 = layers.Conv2D(filters, 3, padding='same', dilation_rate=12, activation='relu')
        self.conv_dil_18 = layers.Conv2D(filters, 3, padding='same', dilation_rate=18, activation='relu')
        self.conv_out = layers.Conv2D(filters, 1, padding='same', activation='relu')

        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.global_max_pool = layers.GlobalMaxPooling2D()
        self.channel_dense1 = layers.Dense(filters // reduction, activation='relu')
        self.channel_dense2 = layers.Dense(filters, activation='sigmoid')

        self.edge_conv1 = layers.Conv2D(filters // 2, 3, padding='same', activation='relu')
        self.edge_conv2 = layers.Conv2D(filters // 2, 3, padding='same', activation='relu')
        self.edge_conv3 = layers.Conv2D(filters, 1, padding='same', activation='sigmoid')

        self.fuse_gate_dense = layers.Dense(filters, activation='sigmoid')

        self.spatial_conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

        self.final_conv = layers.Conv2D(filters, 3, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

        self.residual_conv = layers.Conv2D(filters, 1, padding='same')

    def call(self, x):
        residual = x

        x1 = self.conv1(x)
        x6 = self.conv_dil_6(x)
        x12 = self.conv_dil_12(x)
        x18 = self.conv_dil_18(x)
        x_aspp = self.conv_out(tf.concat([x1, x6, x12, x18], axis=-1))

        avg_pool = self.global_avg_pool(x_aspp)
        max_pool = self.global_max_pool(x_aspp)
        channel_att = self.channel_dense2(self.channel_dense1(avg_pool)) + \
                      self.channel_dense2(self.channel_dense1(max_pool))
        channel_att = tf.expand_dims(tf.expand_dims(channel_att, 1), 1)
        x_channel = x_aspp * channel_att

        edge_feat = self.edge_conv1(x)
        edge_feat = self.edge_conv2(edge_feat)
        edge_feat = self.edge_conv3(edge_feat)

        fuse_input = tf.reduce_mean(tf.concat([x_channel, edge_feat], axis=-1), axis=[1, 2])
        gate = self.fuse_gate_dense(fuse_input)
        gate = tf.expand_dims(tf.expand_dims(gate, 1), 1)

        fused = x_channel * (1 - gate) + edge_feat * gate

        avg_spatial = tf.reduce_mean(fused, axis=-1, keepdims=True)
        max_spatial = tf.reduce_max(fused, axis=-1, keepdims=True)
        spatial_att = self.spatial_conv(tf.concat([avg_spatial, max_spatial, edge_feat], axis=-1))
        x_spatial = fused * spatial_att

        out = self.final_conv(x_spatial)
        out = self.bn(out)

        if residual.shape[-1] != out.shape[-1]:
            residual = self.residual_conv(residual)

        return self.relu(out + residual)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "reduction": self.reduction})
        return config


def res_block(input_tensor, filters, use_dilated_conv=True, use_separable_conv=False, dropout_rate=0.3):
    x = adaptive_gate(input_tensor, filters)
    if use_dilated_conv:
        x = dilated_conv_block(x, filters)
    if use_separable_conv:
        x = depthwise_separable_conv(x, filters)
    x = adaptive_gate(x, filters)
    x = layers.Dropout(dropout_rate)(x)

    shortcut = layers.Conv2D(filters, 1, padding='same')(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)

    return layers.Activation('relu')(layers.Add()([x, shortcut]))


def single_stream_model(input_shape, use_dilated_conv=True, use_separable_conv=False, dropout_rate=0.3):
    spectral = layers.Input((input_shape[0], input_shape[1], 18))
    slope = layers.Input((input_shape[0], input_shape[1], 2))
    elevation = layers.Input((input_shape[0], input_shape[1], 2))
    ndvi = layers.Input((input_shape[0], input_shape[1], 2))
    ndwi = layers.Input((input_shape[0], input_shape[1], 2))
    bi = layers.Input((input_shape[0], input_shape[1], 2))
    bsi = layers.Input((input_shape[0], input_shape[1], 2))
    aspect = layers.Input((input_shape[0], input_shape[1], 2))
    curvature = layers.Input((input_shape[0], input_shape[1], 2))
    terrain = layers.Input((input_shape[0], input_shape[1], 2))

    x = TopographyAwareGate(64)([
        spectral, slope, elevation, ndvi, ndwi,
        bi, bsi, aspect, curvature, terrain
    ])

    e1 = res_block(x, 64)
    p1 = layers.MaxPooling2D(2)(e1)

    e2 = res_block(p1, 128)
    p2 = layers.MaxPooling2D(2)(e2)

    e3 = res_block(p2, 256)
    p3 = layers.MaxPooling2D(2)(e3)

    e4 = res_block(p3, 512)
    p4 = layers.MaxPooling2D(2)(e4)

    e5 = BottleneckASPP_CBAM_EdgeAware(512)(p4)

    d4 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(e5)
    d4 = res_block(layers.Concatenate()([d4, e4]), 512, use_dilated_conv, use_separable_conv, dropout_rate)

    d3 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(d4)
    d3 = res_block(layers.Concatenate()([d3, e3]), 256, use_dilated_conv, use_separable_conv, dropout_rate)

    d2 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(d3)
    d2 = res_block(layers.Concatenate()([d2, e2]), 128, use_dilated_conv, use_separable_conv, dropout_rate)

    d1 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(d2)
    d1 = res_block(layers.Concatenate()([d1, e1]), 64, use_dilated_conv, use_separable_conv, dropout_rate)

    output = layers.Conv2D(1, 1, activation='sigmoid')(d1)

    return Model(
        inputs=[spectral, slope, elevation, ndvi, ndwi, bi, bsi, aspect, curvature, terrain],
        outputs=output
    )


input_shape = (256, 256, 36)
model = single_stream_model(input_shape)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
