"""The original four-input Conv1D + CBAM model, without architectural changes."""
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Activation, Conv1D, MaxPooling1D, BatchNormalization,
    Dropout, Add, Lambda, Reshape, GlobalAveragePooling1D,
    GlobalMaxPooling1D, Multiply, Concatenate,
)


def channel_attention_1d(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared1 = Dense(channel // ratio, activation='relu',
                    kernel_initializer='he_normal', use_bias=True)
    shared2 = Dense(channel, kernel_initializer='he_normal', use_bias=True)

    avg = GlobalAveragePooling1D()(input_feature)
    avg = Reshape((1, channel))(avg)
    avg = shared2(shared1(avg))

    mx  = GlobalMaxPooling1D()(input_feature)
    mx  = Reshape((1, channel))(mx)
    mx  = shared2(shared1(mx))

    feat = Activation('sigmoid')(Add()([avg, mx]))
    return Multiply()([input_feature, feat])


def spatial_attention_1d(input_feature):
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=2, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: tf.reduce_max(x,  axis=2, keepdims=True))(input_feature)
    concat   = Concatenate(axis=2)([avg_pool, max_pool])
    feat     = Conv1D(1, 7, padding='same', activation='sigmoid',
                      kernel_initializer='he_normal', use_bias=False)(concat)
    return Multiply()([input_feature, feat])


def cbam_block_1d(input_feature, residual=False):
    out = channel_attention_1d(input_feature)
    out = spatial_attention_1d(out)
    if residual:
        out = Add()([input_feature, out])
    return out


def build_model(temporal_dim=1000, num_ch=22, num_classes=4):
    """Build the notebook model; inputs are (batch, time, EEG channels)."""
    input_main  = Input(shape=(temporal_dim, num_ch), name="input_main")
    input_theta = Input(shape=(temporal_dim, num_ch), name="input_theta")
    input_alpha = Input(shape=(temporal_dim, num_ch), name="input_alpha")
    input_beta  = Input(shape=(temporal_dim, num_ch), name="input_beta")

    # Concatenate broadband + band-specific inputs along channel axis
    fused = Concatenate(axis=-1)([input_main, input_theta, input_alpha, input_beta])  # (N, 1000, 88)

    x = Conv1D(32, 5, activation='relu', padding='same')(fused)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # CBAM attention over 1-D feature maps
    x = cbam_block_1d(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)

    model = Model(
        inputs  = [input_main, input_theta, input_alpha, input_beta],
        outputs = x,
        name    = "BCI_Multiband_CBAM"
    )
    return model
