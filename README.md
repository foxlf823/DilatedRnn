# A PyTorch implement of Dilated RNN

This code implements the models described in the paper "[Dilated Recurrent Neural Networks](https://arxiv.org/abs/1710.02224)". We refer to [their tensorflow implement](https://github.com/code-terminator/DilatedRnn).

Requirements
-----
* Python 2.7
* PyTorch 0.3

Setup
-----
1. Download the MNIST data via "torchvision.datasets.MNIST" and make the following configuration at "demo.py".

  ```python
  data_dir = '/home/fox/MNIST_data' # the path of MNIST data
  n_steps = 28 # the step of RNN
  input_dims = 28 # the input dimension of each RNN cell
  n_classes = 10 # MNIST has ten classes for classifying
  cell_type = "LSTM" # only support LSTM
  hidden_structs = [20, 20] # Give a list of the dimension in each layer
  dilations = [1, 2] # Give a list of the dilation in each layer
  ```

2. Run "demo.py" and you'll see:

  ```
  ==> Building a dRNN with LSTM cells
  Iter 1, Step 100, Avarage Loss: 1.310388
  Iter 1, Step 200, Avarage Loss: 0.929560
  Iter 1, Step 300, Avarage Loss: 0.943462
  Iter 1, Step 400, Avarage Loss: 0.843424
  ========> Validation Accuarcy: 0.753500
  ...
  ```

Notes
-----
Currently, it only supports CPU and LSTM. Moreover, when the dilation starts not at 1, the output will be fused in the paper. This is not implemented now.

