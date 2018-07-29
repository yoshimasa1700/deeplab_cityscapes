import tensorflow as tf
import numpy as np
import cv2
from matplotlib import gridspec
from matplotlib import pyplot as plt

import sys
sys.path.append('cityscapesScripts')

from cityscapesscripts.helpers import labels

class DeeplabModel():
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, frozen_graph_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None

        with open(frozen_graph_path, 'r') as f:
            graph_def = tf.GraphDef.FromString(f.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def inference(self, image):
        width, height, channels = image.shape
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * height), int(resize_ratio * width))
        resized_image = cv2.resize(image, target_size)

        cv2.imshow("resized", resized_image)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [resized_image]})
        seg_map = batch_seg_map[0]

        result_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        return result_image, seg_map

    def inference_path(self, image_path):
        image = cv2.imread(image_path)
        cv2.imshow("test", image)
        cv2.waitKey(100)
        return self.inference(image)


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    
    Returns:
    A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3

    return colormap


def label_to_color_image(input_label):
    """Adds color defined by the dataset colormap to the label.
    
    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
    is the color indexed by the corresponding element in the input label
    to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
    map maximum entry.
    """
    if input_label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    # colormap = create_pascal_label_colormap()
    colormap = np.asarray([list(label.color) for label in labels.trainId2label.values()])

    if np.max(input_label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[input_label]


output_index = 0

def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    global output_index
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    # plt.show()
    plt.savefig('{:06d}.png'.format(output_index))
    output_index += 1

LABEL_NAMES = np.asarray([label.name for label in labels.trainId2label.values()])
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


def main():
    frozen_graph_path = "../model/deeplabv3_mnv2_cityscapes_train/frozen_inference_graph.pb"
    model = DeeplabModel(frozen_graph_path)

    import glob
    import os.path

    image_file_list = glob.glob("../data/images_higher_camera/*")
    image_file_list.sort()

    for image_file in image_file_list:
        base, ext = os.path.splitext(image_file)

        if ext != ".png":
            continue

        image, segmap = model.inference_path(image_file)

        vis_segmentation(image, segmap)

if __name__ == "__main__":
    main()
