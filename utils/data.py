import tensorflow as tf
import os
import xml.etree.ElementTree as ET
import numpy as np
from utils.config import config
import matplotlib.pyplot as plt


class Transform:
    """
    长边不超过1000，短边不超过600，bbox坐标对应缩放。
    """
    def __init__(self):
        self.max_size = 1000
        self.min_size = 600

    def preprocess(self, image):
        image = image / 255.0

        H, W, C = image.shape
        scale1 = self.min_size / min(H, W)
        scale2 = self.max_size / max(H, W)
        scale = min(scale1, scale2)
        image = tf.image.resize(image, [int(H * scale), int(W * scale)])

        image = (image - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])
        return image

    def resize_bbox(self, bbox, in_size, out_size):
        bbox = bbox.copy()
        y_scale = float(out_size[0]) / in_size[0]
        x_scale = float(out_size[1]) / in_size[1]
        bbox[:, 0] = y_scale * bbox[:, 0]
        bbox[:, 2] = y_scale * bbox[:, 2]
        bbox[:, 1] = x_scale * bbox[:, 1]
        bbox[:, 3] = x_scale * bbox[:, 3]
        return bbox

    def __call__(self, input_data):
        img, bbox, label = input_data
        H, W, C = img.shape
        img = self.preprocess(img)
        n_H, n_W, n_C = img.shape
        bbox = self.resize_bbox(bbox, (H, W), (n_H, n_W))
        scale = n_H / H
        return img, bbox, label, scale


# 标签对应的名字。
VOC_BBOX_LABEL_NAMES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


class VOCBboxDataset:
    """
    读取VOC数据
    labels: (R,) R代表了一张图里真实物体的数量，值为0,1,2,...,代表前面那个元组里第几个元素。
    """

    def __init__(self, data_dir, split='trainval'):
        # 这个file保存了训练集和验证集的编号(trainval.txt)。
        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        id_ = self.ids[i]
        annotation = ET.parse(os.path.join(self.data_dir, 'Annotations', str(id_) + '.xml'))
        bbox = list()
        label = list()
        for obj in annotation.findall('object'):
            if int(obj.find('difficult').text) == 1:
                continue
            # bounding box 就是真实的框啦
            bndbox_anno = obj.find('bndbox')
            # 减一是为了让坐标以0开始
            bbox.append([int(bndbox_anno.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        image = tf.io.read_file(img_file)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, dtype=tf.float32)

        return image, bbox, label

    __getitem__ = get_example


class Dataset:
    def __init__(self, config):
        self.db = VOCBboxDataset(config.voc_data_dir)
        self.tsf = Transform()

    def __getitem__(self, idx):
        ori_img, bbox, label = self.db.get_example(idx)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = tf.expand_dims(img, axis=0)

        return img, bbox, label, scale

    def __len__(self):
        return len(self.db)


def vis(img, bboxes, labels):
    """
    这个函数用来看看一个样本的图形、bbox、对应的类别，和scale大小。
    """
    img = img.numpy() #[-1,1]
    img = (img * 0.225) + 0.45
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)

    for i in range(len(bboxes)):
        y1 = bboxes[i][0]
        x1 = bboxes[i][1]
        y2 = bboxes[i][2]
        x2 = bboxes[i][3]
        height = y2 - y1
        width = x2 - x1
        ax.add_patch(plt.Rectangle((x1,y1), width, height, fill=False, edgecolor='red', linewidth=2))
        ax.text(x1,y1,VOC_BBOX_LABEL_NAMES[labels[i]],style='italic',bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})

    return ax


if __name__ == '__main__':
    print("jinx")
    config._parse({})
    dataset = Dataset(config)
    print("读取数据中....")

    img, bboxes, labels, scale = dataset[1]
    vis(img, bboxes, labels)

    print(img.shape)
    print(bboxes.shape, 'max=', tf.reduce_max(bboxes).numpy())
    print(labels.shape)

