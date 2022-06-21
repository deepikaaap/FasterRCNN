def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', type=bool, default=True, required=False,
                        help="True/False to save the weights of the model")
    parser.add_argument('--save_dir', type=str, default='/FasterRCNN', required=False,
                        help="Path to save the weights of the model")
    parser.add_argument('--epochs', type=int, default=2, required=False, help="Number of epochs for training the model")
    parser.add_argument('--int_lr', type=float, default=1e-3, required=False,
                        help="The initial learning rate of the model training")
    parser.add_argument('--batch_size', type=int, default=1, required=False, help="The batch size for the training")
    parser.add_argument('--label_format', type=str, default='pascal_voc', required=False,
                        help="The format of the dataset like yolo or pascla_voc")
    parser.add_argument('--dataset_mean', default=[95.71277075, 98.54471269, 93.58283276], required=False,
                        help="per channel mean of the images in the dataset")
    parser.add_argument('--dataset_std', default=[1.07244683, 1.0594797, 1.04724088], required=False,
                        help="per channel std of the images in the dataset")
    parser.add_argument('--min_width', type=int, default=300, required=False,
                        help="The minimum width of the shortest side of the image")
    parser.add_argument('--rpn_checkpoint_path', type=str, default='training_rpn/training_chkpt.ckpt',
                        help="The file name where the RPN weights are to be saved")
    parser.add_argument('--det_checkpoint_path', type=str, default='training_det/training_chkpt.ckpt',
                        help="The file name where the weights are to be saved")
    parser.add_argument('--rpn_weights_path', type=str, default='rpn_training_weights.h5')
    parser.add_argument('--rpn_model_path', type=str, default='rpn_trained_model')
    parser.add_argument('--det_weights_path', type=str, default='det_training_weights.h5')
    parser.add_argument('--det_model_path', type=str, default='det_trained_model')
    parser.add_argument('--tensorboard_logs_path', type=str, default='logs',
                        help="The file name where the tensorboard logs are to be saved")
    parser.add_argument('--backbone_network', type=str, default='resnet50',
                        help="The network that is to be used as the backbone for the end to end detection model")
    parser.add_argument('--network_stride', type=int, default=32,
                        help="Total stride in the backbone network until the last convolution layer")
    parser.add_argument('--anchor_sizes', default=[64, 128, 256, 512], help="The sizes of the anchor boxes")
    parser.add_argument('--aspect_ratios', default=[0.5, 1, 2], help="aspect ratios of the anchor boxes")
    parser.add_argument('--max_positive_anchors', type=int, default=128,
                        help="Maximum number of positive anchors to be trained for one batch")
    parser.add_argument('--max_negative_anchors', type=int, default=128,
                        help="Maximum number of negative anchors to be trained for one batch")
    parser.add_argument('--positive_anc_threshold', type=float, default=0.7,
                        help="IoU threshold for considering an anchor positive")
    parser.add_argument('--negative_anc_threshold', type=float, default=0.3,
                        help="IoU threshold for considering an anchor negative")
    parser.add_argument('--pos_roi_threshold', type=float, default=0.3,
                        help="IoU threshold for considering an ROI positive")
    parser.add_argument('--neg_roi_threshold', type=float, default=0.0,
                        help="Lower IoU threshold for considering an anchor negative")
    parser.add_argument('--n_classes', type=int, default=3, help="Number of classes in the dataset")
    sys.argv = ['-f']

    args = parser.parse_args()

    # Generate anchors for RPN model
    all_anchors = create_anchor_boxes(args).reshape((-1, 4))
    target_size = get_resized_image_shape(min_size=args.min_width)
    anchor_count = len(args.anchor_sizes) * len(args.aspect_ratios)
    # Create RPN model
    shared_network = create_shared_network()
    rpn_model = rpn_head(shared_network, anchor_count)
    rpn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=args.int_lr), loss=[rpn_cls_loss_calc, reg_loss_calc])
    print("Before loading rpn_model")
    print(shared_network.layers[2].get_weights()[0])
    # We perform 4 Step Alternate training of RPN and Fast RCNN heads of the model.
    # Step 1, Let us start off with training the RPN with imagenet pre-trained backbone.
    if os.path.exists('/kaggle/input/rpn-loss-trained-15'):  # TODO: change
        rpn_model.load_weights(os.path.join('/kaggle/input/rpn-loss-trained-15/rpn_training_weights.h5'))
    print("after loading rpn model")
    print(shared_network.layers[2].get_weights()[0])
    # Step 2, Train the detection network Fast RCNN head with imagenet pre-trained backbone
    # and output from trained RPN model
    # No shared layers for now.
    # Creating a Detection network with a Fast RCNN head.
    shared_network = create_shared_network()
    detection_model = faster_rcnn_head(shared_network, args.n_classes)
    # Setup training of the detection network
    detection_model.compile(optimizer=tf.optimizers.Adam(learning_rate=args.int_lr),
                            loss=[det_cls_loss_calc, det_reg_loss_calc])
    print("Before Det model")
    print(shared_network.layers[2].get_weights()[0])
    if os.path.exists('/kaggle/input/det-loss-trained-15'):  # TODO: change
        rpn_model.load_weights(os.path.join('/kaggle/input/det-loss-trained-15/det_training_weights.h5'))
        train_detection_network(args, rpn_model, detection_model, all_anchors, continue_epoch=0)
    else:
        train_detection_network(args, rpn_model, detection_model, all_anchors, continue_epoch=0)
    print("after det model")
    print(shared_network.layers[2].get_weights()[0])

    # print(rpn_model.layers[2].get_weights()[0])


if __name__ == "__main__":
    main()