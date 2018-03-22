import tensorflow as tf
import numpy
import csv
import os
import matplotlib.pyplot as plt
from PIL import Image


DATA_PATH = "2016TrainingData/"
LABEL_PATH = "Training_GroundTruth.csv"
TFRECORD_TRAIN_PATH = './TFRecord_train'
TFRECORD_TEST_PATH = './TFRecord_test'

train_image_nums = 800
test_image_nums = 100

def parse_label(label_path):
    """
    解析图片的标签csv文件，将良性设为0，恶性设为1
    :param label_path: 存有标签的csv文件路径
    :return: 图片id列表和对应标签列表
    """
    label_file_name = os.path.join('.', LABEL_PATH)
    with open(label_file_name) as label_f:
        label_csv = csv.reader(label_f)
        image_id_list = []
        label_list = []
        for row in label_csv:
            image_id_list.append(row[0])
            if row[1] == "benign":
                label_list.append(0)
            else:
                label_list.append(1)
        return image_id_list, label_list


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 按照图片id获得图片的路径
temp_file_name = [x for x in os.listdir(os.path.join('.', DATA_PATH)) if os.path.splitext(x)[1] == '.jpg']
sorted_filename_list = sorted(temp_file_name)
file_name_list = []
for file_str in sorted_filename_list:
    file_name_list.append(os.path.join('./2016TrainingData', file_str))

image_id_list, label_list = parse_label(LABEL_PATH)


# 创建一个writer来写TFRecord文件
train_writer = tf.python_io.TFRecordWriter(TFRECORD_TRAIN_PATH)
test_writer = tf.python_io.TFRecordWriter(TFRECORD_TEST_PATH)

for i in range(len(file_name_list)):
    if i < train_image_nums:
    # 对每一个图像进行resize操作，然后转换为二进制
        img = Image.open(file_name_list[i])
        img = img.resize((224, 224))
        img_raw = img.tobytes()
        # 将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构
        example1 = tf.train.Example(features=tf.train.Features(feature={'label': _int64_feature(label_list[i]),
                                                              'image_id': _bytes_feature(bytes(image_id_list[i], encoding='utf-8')),
                                                              'image_raw': _bytes_feature(img_raw)}))
        print(bytes(image_id_list[i], encoding='utf-8'), label_list[i])
         # 将一个Example写入TFRecord文件
        train_writer.write(example1.SerializeToString())
    else:
        img = Image.open(file_name_list[i])
        img = img.resize((224, 224))
        img_raw = img.tobytes()
        # 将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构
        example2 = tf.train.Example(features=tf.train.Features(feature={'label': _int64_feature(label_list[i]),
                                                              'image_id': _bytes_feature(bytes(image_id_list[i], encoding='utf-8')),
                                                              'image_raw': _bytes_feature(img_raw)}))
        print(bytes(image_id_list[i],encoding='utf-8'), label_list[i])
         # 将一个Example写入TFRecord文件
        test_writer.write(example2.SerializeToString())

train_writer.close()
test_writer.close()



