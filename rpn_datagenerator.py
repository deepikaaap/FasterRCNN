import tensorflow as tf
import albumentations as A
import numpy as np
import bbox_utils
import cv2


# Create Datagenerator to produce one batch of input - 1 image and its corresponding GT
class rpn_datagenerator:
    def __init__(self, max_pos_anc, max_neg_anc, \
                 pos_anc_threshold, neg_anc_threshold, \
                 data, min_width, target_size, label_format, \
                 per_channel_mean, per_channel_std, all_anchors, net='rpn'):
        self.training_images_path, self.training_labels = zip(*data)
        self.img_min_width = min_width
        self.target_size = target_size
        self.label_format = label_format
        self.per_channel_mean = per_channel_mean
        self.per_channel_std = per_channel_std
        self.max_pos_anc = max_pos_anc
        self.max_neg_anc = max_neg_anc
        self.pos_anc_threshold = pos_anc_threshold
        self.neg_anc_threshold = neg_anc_threshold
        self.all_anchors = all_anchors
        self.index = 0
        self.net = net

    def __len__(self):
        return len(self.training_images_path)

    def transform_data(self, imgs, labels):
        transform = A.Compose([
            A.Resize(width=self.target_size[0], height=self.target_size[1]),
            A.Normalize(mean=self.per_channel_mean, std=self.per_channel_std)  # TODO
        ], bbox_params=A.BboxParams(format=self.label_format))

        transformed = transform(image=imgs, bboxes=labels)
        transformed_image = transformed['image']
        transformed_bboxes = np.asarray(transformed['bboxes'])
        return transformed_image, transformed_bboxes


    def find_mapping_anchors_for_groundtruth(self, gt_labels):  # Also RPN output
        """
            Find one anchor box corresponding to each Ground Truth box based on the max IoU
            Inputs:
                gt_labels: nd_array of shape(N_gt, 5) Bounding box coords of GT + class of GT
                anc_boxes: nd_array of shape(N_a, 4) Bounding box coords of anchors
                cfg: a dict of all the necessary config values
            Outputs:
                anchor_delta: nd_array of shape(N_gt, 4) Delta regression values between the GT and corresponding anchor
                anchor_label: nd_array of shape(N_gt, 1) Positive, Negative or Neutral objectification
        """
        def filter_boxes_based_on_count(labels, value, upper_limit=128):
            all_ind = np.where(labels == value)[0]
            if len(all_ind) > upper_limit:
                remove_ind = np.random.choice(list(all_ind), size=(len(all_ind) - upper_limit), replace=False)
                labels[remove_ind] = -1
                # filtered_boxes = np.delete(boxes, remove_ind, axis=0)
                all_ind = np.where(labels == value)[0]

            return all_ind

        def filter_non_neutral_anchors(anchor_rpn_cls_labels):
            return np.where(anchor_rpn_cls_labels != -1)[0]

        N_a, N_gt = len(self.all_anchors), len(gt_labels)
        # Initialize anchor labels array to hold info
        # about the anchor being pos, neg or neutral sample
        anchor_rpn_cls_labels = -1 * np.ones((N_a, 1))
        # Find the IoU between the anchors and GT (N_a x N_gt x 1)
        IoU = bbox_utils.calculate_IoU(self.all_anchors, gt_labels)
        # Find the best anchor that matches each GT
        max_IoU_per_gt = np.amax(IoU, axis=0, keepdims=True)  # 1xN_gt
        max_IoU_per_anc = np.max(IoU, axis=1)  # N_ax1
        argmax_IoU_per_anc = np.argmax(IoU, axis=1)  # N_ax1

        # better than argmax, since takes all occurences of max
        # From the paper, the same groundtruth can map to multiple anchors
        anchors_with_maxIoU_per_gt = np.where(IoU == max_IoU_per_gt)[0]

        # The anchors are considered positive(1), if they have the highest IoU with some Groundtruth
        # Or If they have IoU w.r.t GT >= the threshold for positive anchors
        anchor_rpn_cls_labels[anchors_with_maxIoU_per_gt] = 1
        anchor_rpn_cls_labels[max_IoU_per_anc >= self.pos_anc_threshold] = 1

        # Any non-positive anchor with IoU w.r.t GT < threshold for negative anchors are considered negative(0)
        # If they fall in between they are neutral anchors (-1, default)
        neg_ind = (max_IoU_per_anc < self.neg_anc_threshold) & ((anchor_rpn_cls_labels != 1).flatten())
        anchor_rpn_cls_labels[neg_ind] = 0

        # Filter the number of positive and negative boxes to adhere to the limits
        pos_count = self.max_pos_anc
        filtered_pos_ind = filter_boxes_based_on_count(anchor_rpn_cls_labels, 1, upper_limit=pos_count)

        neg_count = pos_count + self.max_neg_anc - len(filtered_pos_ind)
        filtered_neg_ind = filter_boxes_based_on_count(anchor_rpn_cls_labels, 0, upper_limit=neg_count)

        gt_for_anchor_bboxes = np.zeros((N_a, 4))
        gt_for_anchor_bboxes[filtered_pos_ind, :] = gt_labels[argmax_IoU_per_anc[filtered_pos_ind], :4]

        # Find deltas between the GT and the corresponding anchors
        anchor_delta = bbox_utils.find_deltas_from_boxes(gt_for_anchor_bboxes, self.all_anchors, filtered_pos_ind)

        return anchor_rpn_cls_labels, anchor_delta

    def __getitem__(self, index):
        current_img_path = self.training_images_path[index]
        current_labels = self.training_labels[index]
        current_img = cv2.imread(current_img_path, -1)

        x, y = self.transform_data(current_img, current_labels)
        rpn_cls_labels, deltas = self.find_mapping_anchors_for_groundtruth(y)

        return tf.expand_dims(x, axis=0), [tf.expand_dims(rpn_cls_labels, axis=0), tf.expand_dims(deltas, axis=0)]


