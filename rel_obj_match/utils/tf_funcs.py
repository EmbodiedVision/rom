"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import tensorflow as tf


# --------------------------------------------------
# --- TF Dataset Generator


def get_data_iterator(data, cnfg_training=None, train=False):

    map_func = data.wrapped_generate_input()

    dataset_tf = tf.data.Dataset.from_generator(
        data.generator, tf.int32, tf.TensorShape([])
    )
    dataset_tf = dataset_tf.prefetch(tf.data.AUTOTUNE)
    dataset_tf = dataset_tf.map(map_func)
    if train and cnfg_training is not None:
        dataset_tf = dataset_tf.batch(cnfg_training["batch_size"])
    else:
        dataset_tf = dataset_tf.batch(1)
    iterator = iter(dataset_tf)

    return dataset_tf, iterator


# --------------------------------------------------
# --- Methods for Training


def train_step(net, data, optimizer, params):
    """Trains 'net' on 'example' using 'optimizer' wrt 'params'."""
    with tf.GradientTape() as tape:
        output = net(data)
        losses = net.get_loss(output, data, params)
        # net.summary()
    variables = net.trainable_variables
    gradients = tape.gradient(losses["loss_total"], variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return output, losses


def test_step(net, data, eval_dict, cnfg, params=None):
    output = net(data)
    if params is not None:
        losses = net.get_loss(output, data, params)
    else:
        losses = None  # not needed for final test evaluation
    eval_dict = net.evaluate_batch({**data, **output}, eval_dict, cnfg)
    return output, losses, eval_dict


def weight_losses(loss_list, loss_dict, params):
    loss_total = tf.zeros(())
    for name, loss in loss_list:
        loss_dict["l-nw_" + name] = loss
        name = "w-" + name
        if name in params and not (
            isinstance(params[name], float) and params[name] == 0
        ):
            loss_dict[name.replace("w-", "l_")] = params[name] * loss
            loss_total += params[name] * loss
    return loss_total
