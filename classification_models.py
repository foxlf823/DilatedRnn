import torch
import torch.nn as nn
import drnn


def _rnn_reformat(x, input_dims, n_steps):
    """
    This function reformat input to the shape that standard RNN can take. 
    
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
    """
    # permute batch_size and n_steps
    x_reformat = x.permute(1, 0, 2).contiguous()
    # reshape to (n_steps*batch_size, input_dims)
    x_reformat = x_reformat.view(-1, input_dims)
    # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    x_reformat = torch.chunk(x_reformat, n_steps, 0)

    return x_reformat

class drnn_classification(nn.Module):
    """
    This class construct a multilayer dilated RNN for classifiction.  
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
        hidden_structs -- a list, each element indicates the hidden node dimension of each layer.
        dilations -- a list, each element indicates the dilation of each layer.
        n_classes -- the number of classes for the classification.
        input_dims -- the input dimension.
        cell_type -- the type of the RNN cell, should be in ["RNN", "LSTM", "GRU"].
    
    Outputs:
        pred -- the prediction logits at the last timestamp and the last layer of the RNN.
                'pred' does not pass any output activation functions.
    """
    def __init__(self, hidden_structs, dilations, n_classes, input_dims=1):
    
        super(drnn_classification, self).__init__()
        
        self.multi_dRNN_with_dilations = drnn.multi_dRNN_with_dilations(hidden_structs, dilations, input_dims)
        
        self.linear = nn.Linear(hidden_structs[-1], n_classes)
    
    
    def forward(self, inputs): 
        """
        inputs -- the input for the RNN. inputs should be in the form of
            a list of 'n_steps' tenosrs. Each has shape (batch_size, input_dims)
        """
        
        layer_outputs = self.multi_dRNN_with_dilations.multi_dRNN(inputs)
        
        pred = self.linear(layer_outputs[-1])
        
        return pred