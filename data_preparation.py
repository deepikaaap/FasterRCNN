import os
import _pickle as cPickle
import pickle
import random
import numpy
import cv2


def get_full_training_dataset():
    data_dir = os.path.join("/kaggle/input/kitti-3d-object-detection-dataset")
    training_images_dir = data_dir + os.path.join("/training/image_2")
    training_labels_dir = data_dir + os.path.join("/training/label_2")
    testing_images_dir = data_dir + os.path.join("/testing/image_2")

    training_images_path = []
    training_labels_path = []

    for file in os.listdir(training_images_dir):
        training_images_path.append(os.path.join(training_images_dir, file))
        training_labels_path.append(os.path.join(training_labels_dir, file.split('.png')[0] + '.txt'))

    full_training_dataset = list(zip(training_images_path, training_labels_path))

    return full_training_dataset


def get_dataset_path():
    # Loading existing split data to maintain the same shuffling of data for resuming runs
    if os.path.exists(os.path.join('/kaggle/input/pickled-files/training_dataset_path.pkl')):
        print("Unpickling")
        training_dataset_path = pickle.load(open(r'/kaggle/input/pickled-files/training_dataset_path.pkl','rb'))
        validation_dataset_path = pickle.load(open(r'/kaggle/input/pickled-files/validation_dataset_path.pkl','rb'))
    else:
        full_training_dataset = get_full_training_dataset()
        # Splitting training and validation data
        train_split_end = int(len(full_training_dataset) * 0.9)
        training_dataset_path = full_training_dataset[:train_split_end]
        validation_dataset_path = full_training_dataset[train_split_end:]

        # Shuffling the data
        random.shuffle(training_dataset_path)
        print("pickling")
        if not os.path.isdir('dataset'):
            os.mkdir('dataset')
        cPickle.dump(training_dataset_path, open("dataset/training_dataset_path.pkl", "wb"))
        cPickle.dump(validation_dataset_path, open("dataset/validation_dataset_path.pkl", "wb"))

    return training_dataset_path, validation_dataset_path


def read_relevant_bb_labels(labels_path):
    # Reading relevant bounding boxes
    bbox_labels_per_image = []
    class_map = {'pedestrian': 1, 'car': 2, 'cyclist': 3, 'background': 0}
    for labels_file in labels_path:
        bbox_labels = []
        with open(labels_file) as labels:
            for label in labels:
                split_label = label.split(' ')
                bb_class = split_label[0].lower().strip()
                '''Use only data corresponding to the classes supported for Object Detection by KITTI
                   [cars, pedestrians, cyclists]
                '''
                if bb_class in ['pedestrian', 'car', 'cyclist']:
                    bb_lowerleft = [float(split_label[4]), float(split_label[5])]
                    bb_upperright = [float(split_label[6]), float(split_label[7])]

                    # Use Albumentation at later stages to augment data and the bboxes
                    bbox_labels.append([bb_lowerleft[0],
                                        bb_lowerleft[1],
                                        bb_upperright[0],
                                        bb_upperright[1], class_map[bb_class]])
        bbox_labels_per_image.append(bbox_labels)

    return bbox_labels_per_image


def create_pickled_data():
    # Loading existing split data to maintain the same shuffling of data for resuming runs
    if os.path.exists(os.path.join('/kaggle/input/pickled-files/training_data.pkl')):
        print("Unpickling")
        training_data = pickle.load(open(r'/kaggle/input/pickled-files/training_data.pkl', 'rb'))
        validation_data = pickle.load(open(r'/kaggle/input/pickled-files/validation_data.pkl', 'rb'))
    else:
        training_dataset_path, validation_dataset_path = get_dataset_path()
        training_images_path, training_labels_path = zip(*training_dataset_path)
        validation_images_path, validation_labels_path = zip(*validation_dataset_path)

        # Read training and validation labels
        training_bb_labels = read_relevant_bb_labels(training_labels_path)
        validation_bb_labels = read_relevant_bb_labels(validation_labels_path)
        training_data = list(zip(training_images_path, training_bb_labels))
        validation_data = list(zip(validation_images_path, validation_bb_labels))

        print("pickling")
        if not os.path.isdir('dataset'):
            os.mkdir('dataset')
        cPickle.dump(training_data, open("dataset/training_data.pkl", "wb"))
        cPickle.dump(validation_data, open("dataset/validation_data.pkl", "wb"))


def find_feature_mean_and_std(images_path):
    channel_total = np.zeros(shape=(3,))
    for image_path in images_path:
        image = cv2.resize(cv2.imread(image_path, -1), target_size)
        image_mean_per_channel = np.mean(np.reshape(image, (-1, 3)), axis=0)
        channel_total = channel_total + image_mean_per_channel
    channel_mean = channel_total / len(images_path)

    for image_path in images_path[:1]:
        image = np.reshape(cv2.resize(cv2.imread(image_path, -1), target_size), (-1, 3))
        sq_diff = np.mean(np.square(image - channel_mean), axis=0)
    channel_std = np.sqrt(sq_diff / len(images_path))

    return channel_mean, channel_std