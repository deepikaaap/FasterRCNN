def get_resized_image_shape(width=1242, height=375, min_size=600):
    if width < height:
        resized_width = min_size
        resized_height = int(resized_width * height / width)
    else:
        resized_height = min_size
        resized_width = int(resized_height * width / height)

    return resized_width, resized_height


def get_feature_map_size(img_size, network='vgg16'):
    """
    Calculates the size of the feature map to assist in finding the anchors for OD
    Inputs:
        - img_size: Tuple(int,int) Corresponding to the width and height of the input image
        - stride: Int, Corresponding to the stride with which the filters are moved in the image
    Outputs:
        - feature_map_size: Tuple(int,int) Corresponding to the width and height of the feature map
                            from the last layer of the network
    """
    if network == 'vgg16':
        # Cumulative stride until the last layer of VGGNet-16 is 16
        return int(img_size[0]/16), int(img_size[1]/16)

    elif network == 'resnet50':
        # Cumulative stride until the last layer of ResNet-50 is 32 (Stride 2 at each filter/pool/block:[7,3,1,1,1])
        return int(img_size[0]/32) + 1, int(img_size[1]/32) + 1


def named_logs(names, logs):
    result = {}
    for l in zip(names, logs):
        result[l[0]] = l[1]
    return result


def calculate_mean_losses(epoch_rpn_loss_dict):
    mean_loss = {}
    for key, value in epoch_rpn_loss_dict.items():
        all_losses = np.asarray(value)
        mean_loss[key] = np.mean(all_losses, axis=0)
    return mean_loss

