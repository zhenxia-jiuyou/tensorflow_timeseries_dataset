"""by maye"""

import numpy as np
import tensorflow as tf



def timeseries_dataset_from_dict_tensors(dict_tensors, sequence_len, stride=1):
    num_seqs = (len(dict_tensors) - sequence_len ) // stride + 1
    start_indices = np.arange(0, num_seqs, stride)
    start_indices_ds = tf.data.Dataset.from_tensor_slices(start_indices)
    indices_ds = start_indices_ds.map(lambda start_index:
                                      tf.range(start_index, start_index + sequence_len,))
    return indices_ds.map(lambda indices: {key: tf.gather(tensor, indices) for key, tensor in dict_tensors.items()})







