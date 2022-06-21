import argparse
import sys
import os
import utils
import tensorflow as tf
from tensorflow import keras as keras

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', type=bool, default=True, required=False,
                        help="True/False to save the weights of the model")
    parser.add_argument('--save_dir', type=str, default='/FasterRCNN', required=False,
                        help="Path to save the weights of the model")
    parser.add_argument('--epochs', type=int, default=5, required=False, help="Number of epochs for training the model")
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
    parser.add_argument('--tensorboard_logs_path', type=str, default='logs',
                        help="The file name where the tensorboard logs are to be saved")
    parser.add_argument('--backbone_network', type=str, default='resnet50',
                        help="The network that is to be used as the backbone for the end to end detection model")
    parser.add_argument('--network_stride', type=int, default=32,
                        help="Total stride in the backbone network until the last convolution layer")
    parser.add_argument('--anchor_sizes', default=[128, 256, 512], help="The sizes of the anchor boxes")
    parser.add_argument('--aspect_ratios', default=[0.5, 1, 2], help="aspect ratios of the anchor boxes")
    parser.add_argument('--max_positive_anchors', type=int, default=128,
                        help="Maximum number of positive anchors to be trained for one batch")
    parser.add_argument('--max_negative_anchors', type=int, default=128,
                        help="Maximum number of negative anchors to be trained for one batch")
    parser.add_argument('--positive_anc_threshold', type=float, default=0.7,
                        help="IoU threshold for considering an anchor positive")
    parser.add_argument('--negative_anc_threshold', type=float, default=0.3,
                        help="IoU threshold for considering an anchor negative")
    parser.add_argument('--pos_roi_threshold', type=float, default=0.5,
                        help="IoU threshold for considering an ROI positive")
    parser.add_argument('--neg_roi_threshold', type=float, default=0.0,
                        help="Lower IoU threshold for considering an anchor negative")
    parser.add_argument('--n_classes', type=int, default=3, help="Number of classes in the dataset")
    sys.argv = ['-f']

    args = parser.parse_args()

    # Generate anchors for RPN model
    all_anchors = create_anchor_boxes(args).reshape((-1, 4))
    rpn_weights = os.path.join(args.rpn_weights_path)  # TODO: change
    target_size = utils.get_resized_image_shape(min_size=args.min_width)
    anchor_count = len(args.anchor_sizes) * len(args.aspect_ratios)
    # Create RPN model
    shared_network = create_shared_network()
    rpn_model = rpn_head(shared_network, anchor_count)
    rpn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=args.int_lr), loss=[rpn_cls_loss_calc, reg_loss_calc])
    # We perform 4 Step Alternate training of RPN and Fast RCNN heads of the model.
    # Step 1, is training the RPN with imagenet pre-trained backbone
    if os.path.exists(args.rpn_weights_path):  # TODO: change
        rpn_model.load_weights(rpn_weights)
    elif os.path.exists('/kaggle/input/rpntrained10epochs'):
        rpn_model = keras.models.load_model(os.path.join('/kaggle/input/rpntrained10epochs'),
                                            custom_objects={'rpn_cls_loss_calc': rpn_cls_loss_calc,
                                                            'reg_loss_calc': reg_loss_calc})
        rpn_model = train_rpn(args, rpn_model, all_anchors, continue_epoch=3)
    else:
        rpn_model = train_rpn(args, rpn_model, all_anchors)



if __name__ == "__main__":
    main()

