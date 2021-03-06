import pytest
from keras.utils.test_utils import layer_test
from keras.layers.embeddings import Embedding


def test_embedding():
    layer_test(Embedding,
               kwargs={'output_dim': 4., 'input_dim': 10, 'input_length': 2},
               input_shape=(3, 2),
               input_dtype='int32')


if __name__ == '__main__':
    pytest.main([__file__])
