from max.dtype import DType
from max.tensor import Tensor
from max.nn import Embedding

embedding = Embedding(vocab_size=1000, dim=128)
tokens = Tensor.ones([10], dtype=DType.uint64)
embedded = embedding(tokens)
print (embedded)
assert embedded.shape == [10, 128]
