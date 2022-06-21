def calculate_det_validation_loss_after_a_batch_of_training(args, rpn_model, det_model, target_size, all_anchors):
    det_val_datagen = iter(detector_datagenerator(training_data[1:4], args.min_width, target_size, \
                                                  args.label_format, args.dataset_mean, args.dataset_std, \
                                                  all_anchors, rpn_model, args.pos_roi_threshold,
                                                  args.neg_roi_threshold))
    val_loss = []
    val_data_size = len(validation_data)
    for batch_ind in range(val_data_size):
        valX, valY = det_val_datagen.__getitem__(batch_ind)
        val_logs = det_model.test_on_batch(valX, valY)
        val_loss.append(val_logs)

    val_loss_tensor = tf.constant(val_loss, dtype=tf.float32)
    batch_val_loss = tf.reduce_mean(val_loss_tensor, 0)
    return batch_val_loss.numpy()


def train_detection_network(args, rpn_model, detection_model, all_anchors, continue_epoch=0):
    # Setting up callbacks
    checkpoint_path = os.path.join(args.det_checkpoint_path)
    tensorboard_path = os.path.join(args.tensorboard_logs_path)
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=tensorboard_path,
        histogram_freq=0,
        batch_size=len(training_data[:3]),
        write_graph=True,
        write_grads=True
    )
    tensorboard_cb.set_model(detection_model)
    det_train_names = ['det_train_tot_loss', 'det_train_cls_loss', 'det_train_reg_loss']
    det_val_names = ['det_val_tot_loss', 'det_val_cls_loss', 'det_val_reg_loss']
    training_data_size = len(training_data[:3])

    # Creating the batch of data to run through the model.
    for epoch in tqdm(range(args.epochs)):
        epoch_pbar = Progbar(training_data_size, unit_name='batch')
        # Generate Datagenerators for training and validation
        det_train_datagen = detector_datagenerator(training_data[5:8], args.min_width, target_size, \
                                                   args.label_format, args.dataset_mean, args.dataset_std, \
                                                   all_anchors, rpn_model, args.pos_roi_threshold,
                                                   args.neg_roi_threshold)

        for batch_ind in range(training_data_size):
            X, y = det_train_datagen.__getitem__(batch_ind)
            train_logs = detection_model.train_on_batch(X, y)
            val_logs = calculate_det_validation_loss_after_a_batch_of_training(args, rpn_model, detection_model,
                                                                               target_size, all_anchors)

            if batch_ind == (training_data_size - 1):
                # rpn_model.save_weights(checkpoint_path)
                detection_model.save_weights(os.path.join(args.det_weights_path))
                detection_model.save(os.path.join(args.det_model_path), save_format='tf')

            # Find the outputs of the RPN from
            epoch_pbar.update(batch_ind,
                              values=[("det_train_cls_loss", train_logs[1]), ("det_train_reg_loss", train_logs[2]),
                                      ("det_val_cls_loss", val_logs[1]), ("det_val_reg_loss", val_logs[2])])

        tensorboard_cb.on_epoch_end(epoch, named_logs(det_train_names, mean_losses['train']))
        tensorboard_cb.on_epoch_end(epoch, named_logs(det_val_names, mean_losses['val']))

    tensorboard_cb.on_train_end(None)