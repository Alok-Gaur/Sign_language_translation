import os
import sys
import numpy as np
import tensorflow as tf


def load_numpy(file_path):
    data = np.load(file_path.numpy().decode('utf-8'))
    return data.astype(np.float32)

def create_dataset(data_dir):
    class_names = sorted(os.listdir(data_dir))
    class_to_index = {clss:i for i, clss in enumerate(class_names)}
    file_paths, labels = [], []

    for clss in class_names:
        class_path = os.path.join(data_dir, clss)
        
        if not os.path.isdir(class_path):
            continue

        for fname in os.listdir(class_path):
            if fname.endswith(".npy"):
                file_paths.append(os.path.join(class_path,fname))
                labels.append(class_to_index[clss])
    
    path_tensor = tf.constant(file_paths)
    label_tensor = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((path_tensor, label_tensor))

    def map_func(fp, label):
        landmark = tf.py_function(load_numpy, inp=[fp], Tout=tf.float32)
        landmark.set_shape([63])
        return landmark, tf.one_hot(label, depth=len(class_names))
    return dataset.map(map_func).shuffle(200).batch(16), class_names
