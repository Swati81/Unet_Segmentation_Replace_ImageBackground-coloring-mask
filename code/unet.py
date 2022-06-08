from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import concatenate, Conv2DTranspose, Dropout


# unet-models with Keras Functional API
def double_conv_block(x, n_filters):
    x = Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = MaxPooling2D(2)(f)
    p = Dropout(0.4)(p)
    return f, p


def upsample_block(x, conv_features, n_filters):
    x = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = concatenate([x, conv_features])
    x = Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x


# ############## multiclass u-net models ###########################################
def unet_model(size=(256, 256, 3), n_class=3):
    inputs = Input(shape=size)
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)

    bottleneck = double_conv_block(p4, 1024)

    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)
    outputs = Conv2D(n_class, 1, padding="same", activation="softmax")(u9)
    model = Model(inputs, outputs, name="U-Net")
    return model


def unet_model_s(size=(256, 256, 3), n_class=3):
    inputs = Input(shape=size)
    f1, p1 = downsample_block(inputs, 32)
    f2, p2 = downsample_block(p1, 64)
    f3, p3 = downsample_block(p2, 128)
    f4, p4 = downsample_block(p3, 256)

    bottleneck = double_conv_block(p4, 512)

    u6 = upsample_block(bottleneck, f4, 256)
    u7 = upsample_block(u6, f3, 128)
    u8 = upsample_block(u7, f2, 64)
    u9 = upsample_block(u8, f1, 32)
    outputs = Conv2D(n_class, 1, padding="same", activation="softmax")(u9)

    model = Model(inputs, outputs, name="U-Net-s")
    return model


def simple_unet_model(size=(256, 256, 3), n_class=3):
    s = Input(size)
    # Contraction
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.30)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.30)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.30)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.30)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    # bottleneck
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.30)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    # Extraction
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.30)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.30)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.30)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.30)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    outputs = Conv2D(n_class, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[s], outputs=[outputs])

    return model


def medium_unet_model(size=(256, 256, 3), n_class=3):
    inputs = Input(size)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c4)
    d4 = Dropout(0.5)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(d4)

    c5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c5)
    d5 = Dropout(0.5)(c5)

    up6 = Conv2D(256, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(d5))
    merge6 = concatenate([d4, up6], axis=3)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c6)

    up7 = Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c6))
    merge7 = concatenate([c3, up7], axis=3)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    c7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c7)

    up8 = Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c7))
    merge8 = concatenate([c2, up8], axis=3)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c8)

    up9 = Conv2D(32, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c8))
    merge9 = concatenate([c1, up9], axis=3)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    c9 = Conv2D(3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    out = Conv2D(n_class, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[out])

    return model


# ############## binary u-net models ##########################################################
def unet_binary(size=(256, 256, 1)):
    inputs = Input(size)
    c1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    c2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    c3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    c4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c4)
    d4 = Dropout(0.5)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(d4)

    c5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p4)
    c5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c5)
    d5 = Dropout(0.5)(c5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(d5))
    merge6 = concatenate([d4, up6], axis = 3)
    c6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    c6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c6))
    merge7 = concatenate([c3, up7], axis=3)
    c7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    c7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c7))
    merge8 = concatenate([c2, up8], axis = 3)
    c8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    c8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c8))
    merge9 = concatenate([c1, up9], axis=3)
    c9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    c9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    c9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    out = Conv2D(3, 3, activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[out])

    return model


def unet_binary_medium(size=(256, 256, 1)):
    inputs = Input(size)
    c1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    c2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    c3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    c4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    c4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c4)
    d4 = Dropout(0.5)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(d4)

    c5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p4)
    c5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c5)
    d5 = Dropout(0.5)(c5)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(d5))
    merge6 = concatenate([d4, up6], axis = 3)
    c6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    c6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c6))
    merge7 = concatenate([c3, up7], axis=3)
    c7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    c7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c7))
    merge8 = concatenate([c2, up8], axis=3)
    c8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    c8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c8))
    merge9 = concatenate([c1, up9], axis=3)
    c9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    c9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    c9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    out = Conv2D(1, 1, activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[out])

    return model
