import tensorflow as tf
import numpy as np


def x1y1x2y2_to_xywh_format(boxes, batch=False):
    if batch:
        w = boxes[:, :, 2] - boxes[:, :, 0]
        h = boxes[:, :, 3] - boxes[:, :, 1]
        x = boxes[:, :, 0] + (0.5 * w)
        y = boxes[:, :, 1] + (0.5 * h)
    else:
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        x = boxes[:, 0] + (0.5 * w)
        y = boxes[:, 1] + (0.5 * h)

    return x, y, w, h


def find_deltas_from_boxes(gt_for_anchor_bboxes, anc_boxes, positive_anc_ind):
    '''
    Used for finding the parameterization for the regression function of the RPN.
    The coords are (x_center,y_center, width, height)
    tx = (x - x_a)/w_a
    ty = (y - y_a)/h_a
    tw = log(w/w_a)
    th = log(h/h_a)

    Output Format : [tx,ty,tw,th]
    '''
    if gt_for_anchor_bboxes.shape[1] == 4:
        delta = np.zeros((len(anc_boxes), 4))
    else:
        delta = np.zeros((len(anc_boxes), 5))

    anc_boxes = anc_boxes[positive_anc_ind, :]
    gt_for_anchor_bboxes2 = gt_for_anchor_bboxes[positive_anc_ind, :4]

    x_a, y_a, w_a, h_a = x1y1x2y2_to_xywh_format(anc_boxes)
    x, y, w, h = x1y1x2y2_to_xywh_format(gt_for_anchor_bboxes2)

    delta[positive_anc_ind, 0] = (x - x_a) / w_a
    delta[positive_anc_ind, 1] = (y - y_a) / h_a
    delta[positive_anc_ind, 2] = np.log(w / w_a)
    delta[positive_anc_ind, 3] = np.log(h / h_a)
    if delta.shape[1] == 5:
        delta[positive_anc_ind, 4] = gt_for_anchor_bboxes[positive_anc_ind, 4]
    return delta


def find_bbox_from_rpn_deltas(anc_boxes, rpn_deltas):
    '''
    Inverse Mapping from deltas to bboxes.
    x = tx*w_a + x_a
    y = ty*h_a + y_a
    w = exp(tw)*w_a
    h = exp(th)*h_a

    Inputs:
        anc_boxes: shape = (N_a , 4) = (N_a , [x1,y1,x2,y2])
        rpn_deltas: shape = (batch_size, f_h, f_w, N_a*4) = (batch_size, f_h, f_w, N_a*[tx, ty, tw,th])
    Outputs:
        rpn_bboxes: (batch_size, N_a, 4) = (batch_size, N_a, [x1,y1,x2,y2])
    '''
    x_a, y_a, w_a, h_a = x1y1x2y2_to_xywh_format(np.expand_dims(anc_boxes, axis=0), batch=True)
    # rpn_deltas = tf.squeeze(rpn_deltas, axis=0) # TODO: Fix the batchsize incompatibility
    rpn_deltas = tf.reshape(rpn_deltas, (tf.shape(rpn_deltas)[0], len(anc_boxes), 4))
    x = rpn_deltas[:, :, 0] * w_a + x_a
    y = rpn_deltas[:, :, 1] * h_a + y_a
    w = tf.exp(rpn_deltas[:, :, 2]) * w_a
    h = tf.exp(rpn_deltas[:, :, 3]) * h_a

    x1 = x - 0.5 * w
    y1 = y - 0.5 * h
    x2 = x + 0.5 * w
    y2 = y + 0.5 * h

    return tf.stack([x1, y1, x2, y2], axis=-1)  # or axis= -1 last


def find_nmsed_roi(rpn_roi_bboxes, rpn_pred_labels, nms_threshold=0.7, maxN_op_boxes=300):
    '''
    Using the combined_non_max_suppression method from tensorflow "https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression"
    For RPN, there is no class specific box suppression, so the number of classes in the used function is set to 1.
    Inputs:
        bboxes: 4D tensor, shape: [batch_size, N_boxes, 1, [x1,y1,x2,y2]] - The third dim is 1, since we dont do class specific suppression
        bbox_labels: 3D tensor, shape [batch_size, N_boxes, 1] - Classification op from RPN,i.e. confidence scores
        nms_threshold: Boxes having IoU more than this value with the selected boxes are suppressed
        maxN_op_boxes: Maximum number of boxes to be output from the operation
    Outputs:
        nmsed_boxes: 3D tensor, shape: [batch_size, maxN_op_boxes, [y1,x1,y2,x2]]
    '''

    rpn_roi_bboxes = tf.expand_dims(rpn_roi_bboxes, axis=2)
    rpn_pred_labels = tf.reshape(rpn_pred_labels, (tf.shape(rpn_pred_labels)[0], -1, 1))
    # bboxes = tf.transpose(bboxes, [1,0,3,2])
    nmsed_rois, _, _, _ = tf.image.combined_non_max_suppression(rpn_roi_bboxes, rpn_pred_labels, maxN_op_boxes,
                                                                maxN_op_boxes, iou_threshold=nms_threshold,
                                                                clip_boxes=False)
    return nmsed_rois


def calculate_IoU(box1, box2):
    '''
    Inputs:
        box1: shape = (N_gt, 4)
        box2: shape = (N_anc, 4)
    Outputs:
        IoU: shape = (N_gt, N_anc, 1)
    '''

    def area_of_bb(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    if tf.is_tensor(box1):
        box1 = box1.numpy()

    anc_x = box1[:, [0, 2]].reshape(1, -1, 2).transpose(1, 0, 2).astype('float64')
    anc_y = box1[:, [1, 3]].reshape(1, -1, 2).transpose(1, 0, 2).astype('float64')

    gt_x = box2[:, [0, 2]].reshape(1, -1, 2).astype('float64')
    gt_y = box2[:, [1, 3]].reshape(1, -1, 2).astype('float64')

    # Find the max along the x and y co-ords of all the combinations of the boxes
    max_x = np.maximum(gt_x, anc_x)
    max_y = np.maximum(gt_y, anc_y)

    # Find the min along the x and y co-ords of all the combinations of the boxes
    min_x = np.minimum(gt_x, anc_x)
    min_y = np.minimum(gt_y, anc_y)

    # To find the leftmost and rightmost points of the intersecting area
    # We need to find the min of the max x,y co-ords and max of the min x,y co-ords
    left_x = max_x.min(axis=2)
    right_x = min_x.max(axis=2)
    bottom_y = max_y.min(axis=2)
    top_y = min_y.max(axis=2)

    # The width and height are -ve if there is no overlap
    # Defaulting such cases to zero
    int_width = np.maximum(np.zeros(right_x.shape), right_x - left_x)
    int_height = np.maximum(np.zeros(top_y.shape), top_y - bottom_y)

    intersection_area = int_height * int_width

    # Union of the two boxes is the area of the two boxes - the area of intersection
    area_box1 = area_of_bb(box1[:, :4].astype('float64'))  # shape: (N_gt,1)
    area_box2 = area_of_bb(box2[:, :4].astype('float64'))  # shape: (N_anc,1)

    Union = np.expand_dims(area_box1, axis=1) + np.expand_dims(area_box2, axis=0) - intersection_area
    IoU = intersection_area / Union

    return IoU


