import tensorflow as tf
from keras import Input,initializers
from keras.layers import Conv2D, Dense, MaxPooling2D, TimeDistributed, Flatten, Dropout, Layer
from keras import Model


class RPN_Pooling_Layer(Layer):
    '''
    Fast RCNN detectors - ROI Pooling: special case of Spatial Pyramid Pooling
    Considering batch size 1
    Pooling is done by bilinear interpolation crop as opposed to maxpooling in the paper
    '''

    def __init__(self, channel_size, N_pooling_regions, **kwargs):
        super(RPN_Pooling_Layer, self).__init__(**kwargs)
        self.channel_size = channel_size
        self.N_pooling_regions = N_pooling_regions

    def __call__(self, inputs):
        shared_network_output = inputs[0]
        input_rois_from_rpn = tf.reshape(inputs[1], (-1, 4))

        # The batch size is 1, so box_ind = 0 for all the anchors, since we have only 1 image
        box_ind = tf.zeros(tf.shape(input_rois_from_rpn)[0], dtype=tf.int32)
        pooled_images = tf.image.crop_and_resize(shared_network_output, input_rois_from_rpn, box_ind,
                                                 (self.N_pooling_regions, self.N_pooling_regions))
        pooled_images = tf.expand_dims(pooled_images, axis=0)

        # Just to view the projected anchor box on the feature map created by the pooling layer
        # show_image_with_bbox(pooled_images[534,:,:,1045:1048].numpy(), anc_boxes[4:5]/32)
        return pooled_images


def faster_rcnn_head(shared_network, n_classes, roi_bbox_tensor=None, roi_bbox_tensor_shape=[None, None, 4]):
    # RPN ROI BBox layer
    # rpn_roi_bboxes = RPN_ROI_bbox_Layer(all_anchors, name='ROI_bbox')([rpn_pred_labels, rpn_pred_deltas])
    if roi_bbox_tensor is None:
        rpn_roi_bboxes = Input(shape=roi_bbox_tensor_shape)
    else:
        input_img = Input(tensor=roi_bbox_tensor, shape=roi_bbox_tensor_shape)
    # RoiPooling layer
    # Spatial Pooling to 7x7 to match the input to the first fully connected layer of the ResNet
    N_pooling_regions = 7
    channel_size = shared_network.output_shape[3]
    roi_pooling_op = RPN_Pooling_Layer(channel_size, N_pooling_regions, name='ROI_pooling')(
        [shared_network.output, rpn_roi_bboxes])

    # RoI pooling layer is followed by two fully connected layers
    # Hyper-parameters from Spatial Pyramid Pooling Paper
    # We use TimeDistributed because it uses the same weights simultaneously on all the RoIs,
    # instead of applying them separately if each RoI is considered a separate instance
    op = TimeDistributed(Flatten(), name="FlatteningLayer")(roi_pooling_op)
    op = TimeDistributed(Dense(4096, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)),
                         name='fc1')(op)
    op = TimeDistributed(Dropout(0.2), name='DropOut1')(op)
    op = TimeDistributed(Dense(4096, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)),
                         name='fc2')(op)
    op = TimeDistributed(Dropout(0.2), name='DropOut2')(op)

    # Followed by K+1 softmax layer for classification and Regression layer
    # Hyper-paramaters from Fast RCNN paper
    cls_op = TimeDistributed(
        Dense(n_classes + 1, activation='softmax', kernel_initializer=initializers.RandomNormal(stddev=0.01)),
        name='Classification_layer')(op)
    reg_op = TimeDistributed(
        Dense(n_classes * 4, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001)),
        name='Regression_layer')(op)

    detection_model = Model(inputs=[shared_network.input, rpn_roi_bboxes], outputs=[cls_op, reg_op])

    return detection_model