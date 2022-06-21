import tensorflow as tf
from tensorflow import keras as keras
from keras import Input, initializers
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, Dropout, Layer
from keras import Model


# Creating a ResNet50 backbone
def create_shared_network(input_tensor=None, input_shape=(None, None, 3), step=2, weights_path=None):
    if input_tensor is None:
        input_img = Input(shape=input_shape)
    else:
        input_img = Input(tensor=input_tensor, shape=input_shape)
    if step <= 2:
        backbone_network = ResNet50(include_top=False, weights='imagenet', input_tensor=input_img,
                                    input_shape=input_shape)
    else:
        backbone_network = ResNet50(include_top=False, weights=weights_path, input_tensor=input_img,
                                    input_shape=input_shape)

    return backbone_network


def rpn_head(shared_network, n_anc):
    op = Conv2D(512, (3, 3), padding='same', activation='relu', name='rpn_conv1',
                kernel_initializer=initializers.RandomNormal(stddev=0.01))(shared_network.output)

    op_cls = Conv2D(n_anc, (1, 1), activation='sigmoid', name='rpn_cls',
                    kernel_initializer=initializers.RandomNormal(stddev=0.01))(op)
    op_reg = Conv2D(n_anc * 4, (1, 1), activation='linear', name='rpn_reg',
                    kernel_initializer=initializers.RandomNormal(stddev=0.01))(op)

    rpn_model = Model(inputs=shared_network.input, outputs=[op_cls, op_reg])

    return rpn_model


def rpn_cls_loss_calc(*args):
    y_gt, y_pred = args
    y_gt = tf.reshape(y_gt, (1, -1, 1))
    y_pred = tf.reshape(y_pred, (1, -1, 1))

    # Remove gt with value -1 and corresponding preds. We need only 1/0 values of GT
    true_ind = tf.where(y_gt != tf.constant(-1.0, dtype=tf.float32))
    y_gt = tf.gather_nd(y_gt, true_ind)
    y_pred = tf.gather_nd(y_pred, true_ind)
    num_samples = tf.cast(tf.shape(y_gt)[0], dtype=tf.float32)
    bc_loss = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

    return bc_loss(y_gt, y_pred) / num_samples


def reg_loss_calc(*args):
    y_gt, y_pred = args

    y_gt = tf.reshape(y_gt, (1, -1, 4))
    y_pred = tf.reshape(y_pred, (1, -1, 4))

    num_samples = tf.cast(tf.shape(y_gt)[1], dtype=tf.float32)
    lmbda = tf.cast(10, dtype=tf.float32)
    true_mask = tf.where(tf.not_equal(tf.math.count_nonzero(y_gt, axis=2), tf.constant([0], dtype=tf.int64)))
    y_gt = tf.gather_nd(y_gt, true_mask)
    y_pred = tf.gather_nd(y_pred, true_mask)
    h_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    reg_loss = h_loss(y_gt, y_pred)
    reg_loss = tf.reduce_sum(reg_loss, axis=-1)

    return lmbda * reg_loss / num_samples