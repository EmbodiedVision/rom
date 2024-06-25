"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers
import tensorflow.keras.backend as K
from classification_models import Classifiers as Classifiers_addons


def load_encoder(backbone_encoder_net, img_shape):
    """

    :param backbone_encoder_net:
    :param img_shape:
    :return:
    """
    encoder_model = None
    preprocess_input_layer = None

    if backbone_encoder_net == "ResNet50":
        encoder_model = tf.keras.applications.resnet50.ResNet50(
            input_shape=img_shape + (3,), include_top=False, weights="imagenet"
        )
    elif backbone_encoder_net == "ResNet34":
        ResNet34, preprocess_input_layer = Classifiers_addons.get("resnet34")
        encoder_model = ResNet34(
            input_shape=img_shape + (3,), weights="imagenet", include_top=False
        )

    elif backbone_encoder_net == "VGG16":
        encoder_model = tf.keras.applications.vgg16.VGG16(
            input_shape=img_shape + (3,), include_top=False, weights="imagenet"
        )
    elif backbone_encoder_net == "MobileNetV2":
        encoder_model = tf.keras.applications.MobileNetV2(
            input_shape=img_shape + (3,), include_top=False, weights="imagenet"
        )
    elif backbone_encoder_net == "EfficientNetB0":
        encoder_model = tf.keras.applications.efficientnet.EfficientNetB0(
            input_shape=img_shape + (3,), include_top=False, weights="imagenet"
        )
    elif backbone_encoder_net == "EfficientNetB3":
        encoder_model = tf.keras.applications.efficientnet.EfficientNetB3(
            input_shape=img_shape + (3,), include_top=False, weights="imagenet"
        )
    elif backbone_encoder_net != "use_pretrained":
        print(
            "[ERROR] romnet.load_encoder(): Backbone Encoder Network {} unknown.".format(
                backbone_encoder_net
            )
        )
        exit()

    return encoder_model, preprocess_input_layer


def create_mlp(
    hidden_units,
    batch_normaliztion=True,
    dropout_rate=0.2,
    reg_wght=0.0,
    activation_last_layer=False,
    name=None,
):

    mlp_layers = []

    for i, units in enumerate(hidden_units):
        if batch_normaliztion and i < len(hidden_units) - 1:
            mlp_layers.append(layers.BatchNormalization())
        if dropout_rate is not None and i < len(hidden_units) - 1:
            mlp_layers.append(layers.Dropout(dropout_rate))

        mlp_layers.append(layers.Dense(units))

        if i < len(hidden_units) - 1 or activation_last_layer:
            mlp_layers.append(layers.Activation(tf.nn.gelu))

    return tf.keras.Sequential(mlp_layers, name=name)


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


class AttentionalPropagation(layers.Layer):
    def __init__(self, dim_feat, num_heads):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim_feat)
        self.head = create_mlp(
            [2 * dim_feat, 2 * dim_feat, dim_feat],
            batch_normaliztion=False,
            dropout_rate=None,
        )

    def call(self, x, source):
        message = self.mha(query=x, value=source, key=source)
        return self.head(tf.concat([x, message], axis=-1))


class AttentionalGNN(layers.Layer):

    def __init__(self, dim_feat, layers_type):
        super().__init__()

        num_heads = 4
        self.layers = [
            AttentionalPropagation(dim_feat, num_heads) for _ in range(len(layers_type))
        ]
        self.layer_type = layers_type  # 'cross' or 'self'

    def call(self, feats0, feats1):
        """
        :param feats0:  (BS, N_obj_1, D)
        :param feats1:  (BS, N_obj_2, D)
        :return:        [(BS, N_obj_1, D), (BS, N_obj_2, D)]
        """
        for layer, type in zip(self.layers, self.layer_type):
            if type == "cross":
                src0, src1 = feats1, feats0
            else:
                src0, src1 = feats0, feats1
            delta0 = layer(feats0, src0)
            delta1 = layer(feats1, src1)
            feats0 = feats0 + delta0
            feats1 = feats1 + delta1
        return feats0, feats1


def sinkhorn_iter(Z, log_mu, log_nu, iters):
    u, v = tf.zeros_like(log_mu), tf.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - tf.math.reduce_logsumexp(Z + tf.expand_dims(v, axis=1), axis=2)
        v = log_nu - tf.math.reduce_logsumexp(Z + tf.expand_dims(u, axis=2), axis=1)
    return Z + tf.expand_dims(u, axis=2) + tf.expand_dims(v, axis=1)


def sinkhorn(sim_mat, alpha, iters, def_n_obj=None):
    """
    :param sim_mat:     ([BS,] N_obj1, N_obj2)
    :param alpha:       <tf.float32>
    :param iters:       <int>
    :param def_n_obj:   (<int>, <int>)
    :return:            ([BS,] N_obj1+1, N_obj2+1)
    """

    used_batch_dim = len(sim_mat.shape) == 3
    if not used_batch_dim:
        sim_mat = tf.expand_dims(sim_mat, axis=0)

    b, m, n = sim_mat.shape
    if m is None or n is None:
        m, n = def_n_obj
    sim_mat_new_row = tf.tile(tf.reshape(alpha, (1, 1, 1)), (b, 1, n))
    sim_mat_ext = tf.concat([sim_mat, sim_mat_new_row], axis=1)

    sim_mat_new_column = tf.tile(tf.reshape(alpha, (1, 1, 1)), (b, m + 1, 1))
    sim_mat_ext = tf.concat([sim_mat_ext, sim_mat_new_column], axis=2)  # couplings

    mt, nt = [tf.constant(m, dtype=tf.float32), tf.constant(n, dtype=tf.float32)]
    norm = -tf.math.log(mt + nt)
    log_mu = tf.concat(
        [tf.repeat(norm, repeats=m), tf.expand_dims(tf.math.log(nt) + norm, axis=0)],
        axis=0,
    )
    log_nu = tf.concat(
        [tf.repeat(norm, repeats=n), tf.expand_dims(tf.math.log(mt) + norm, axis=0)],
        axis=0,
    )
    log_mu = tf.tile(tf.expand_dims(log_mu, axis=0), (b, 1))
    log_nu = tf.tile(tf.expand_dims(log_nu, axis=0), (b, 1))

    Z = sinkhorn_iter(sim_mat_ext, log_mu, log_nu, iters)
    Z = Z - norm
    if not used_batch_dim:
        Z = Z[0]
    return Z


def sinkhorn_from_numpy(sim_mat, alpha, iters):

    sim_mat = tf.convert_to_tensor(sim_mat, dtype=tf.float32)
    alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)

    res = sinkhorn(sim_mat, alpha, iters)
    return res
