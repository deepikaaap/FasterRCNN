from tqdm import tqdm
import os
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import rpn_datagenerator as rpn_dg
import utils


def calculate_validation_loss_after_a_batch_of_training(args, rpn_model, target_size, all_anchors):
    rpn_val_datagen = rpn_dg.rpn_datagenerator(args.max_positive_anchors, args.max_negative_anchors, \
                                        args.positive_anc_threshold, \
                                        args.negative_anc_threshold, validation_data, args.min_width, target_size, \
                                        args.label_format, args.dataset_mean, args.dataset_std, all_anchors)
    val_loss = []
    val_data_size = len(validation_data)
    for batch_ind in range(val_data_size):
        valX, valY = rpn_val_datagen.__getitem__(batch_ind)
        val_logs = rpn_model.test_on_batch(valX, valY)
        val_loss.append(val_logs)

    val_loss_tensor = tf.constant(val_loss, dtype=tf.float32)
    batch_val_loss = tf.reduce_mean(val_loss_tensor, 0)
    return batch_val_loss.numpy()


def train_rpn(args, rpn_model, all_anchors, continue_epoch=0):
    # Setting up callbacks
    checkpoint_path = os.path.join(args.rpn_checkpoint_path)
    tensorboard_path = os.path.join(args.tensorboard_logs_path)
    training_data_size = len(training_data)
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=tensorboard_path,
        histogram_freq=0,
        batch_size=training_data_size * args.epochs,
        write_graph=True,
        write_grads=True
    )

    tensorboard_cb.set_model(rpn_model)
    rpn_train_names = ['rpn_train_tot_loss', 'rpn_train_cls_loss', 'rpn_train_reg_loss']
    rpn_val_names = ['rpn_val_tot_loss', 'rpn_val_cls_loss', 'rpn_val_reg_loss']

    # Creating the batch of data to run through the model.
    for epoch in tqdm(range(continue_epoch, continue_epoch + args.epochs)):
        epoch_pbar = Progbar(training_data_size, unit_name='batch')
        # Generate Datagenerators for training and validation
        rpn_train_datagen = rpn_dg.rpn_datagenerator(args.max_positive_anchors, args.max_negative_anchors,
                                              args.positive_anc_threshold, \
                                              args.negative_anc_threshold, training_data, args.min_width, target_size, \
                                              args.label_format, args.dataset_mean, args.dataset_std, all_anchors)

        for batch_ind in range(training_data_size):
            X, y = rpn_train_datagen.__getitem__(batch_ind)
            train_logs = rpn_model.train_on_batch(X, y)
            epoch_batch_ind = (epoch * training_data_size) + batch_ind
            tensorboard_cb.on_epoch_end(epoch_batch_ind, utils.named_logs(rpn_train_names, train_logs))

            if (batch_ind % 50 == 0 or batch_ind == (training_data_size - 1) or batch_ind == 0):
                val_logs = calculate_validation_loss_after_a_batch_of_training(args, rpn_model, target_size,
                                                                               all_anchors)
                epoch_pbar.update(batch_ind,
                                  values=[("rpn_train_cls_loss", train_logs[1]), ("rpn_train_reg_loss", train_logs[2]),
                                          ("rpn_val_cls_loss", val_logs[1]), ("rpn_val_reg_loss", val_logs[2])])

            tensorboard_cb.on_epoch_end(epoch_batch_ind, utils.named_logs(rpn_val_names, val_logs))

        rpn_model.save_weights(os.path.join(args.rpn_weights_path))
        rpn_model.save(os.path.join(args.rpn_model_path), save_format='tf')

    tensorboard_cb.on_train_end(None)

    return rpn_model