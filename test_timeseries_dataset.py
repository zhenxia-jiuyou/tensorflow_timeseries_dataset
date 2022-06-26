
import pytest
import tensorflow as tf

from timeseries_dataset import timeseries_dataset_from_dict_tensors


@pytest.fixture()
def dict_tensors():
    return {'int_feature': tf.constant([1, 2, 3]), 'str_feature': tf.constant(['a', 'b', 'c'])}


def test_series_dataset_from_dict_tensors(dict_tensors):
    sequence_len = 2
    stride = 1
    expected_result = tf.data.Dataset.from_tensor_slices({'int_feature': [[1, 2], [2, 3]],
                                                          'str_feature': [['a', 'b'], ['b', 'c']]})
    result = timeseries_dataset_from_dict_tensors(dict_tensors, sequence_len=sequence_len, stride=stride)

    for element in tf.data.Dataset.zip((result, expected_result)).as_numpy_iterator():
        assert element[0].keys() == element[1].keys()
        assert (element[0]['int_feature'] == element[1]['int_feature']).all()
        assert (element[0]['str_feature'] == element[1]['str_feature']).all()


