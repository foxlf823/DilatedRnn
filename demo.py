
import torch
import torchvision
import torch.autograd as autograd
import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn
import classification_models

# configurations
data_dir = '/home/fox/MNIST_data'
n_steps = 28
input_dims = 28
n_classes = 10 

# model config
cell_type = "LSTM" # only support LSTM
hidden_structs = [20, 20] # Give a list of the dimension in each layer
dilations = [1, 2] # Give a list of the dilation in each layer
assert(len(hidden_structs) == len(dilations))

# learning config
batch_size = 128
learning_rate = 1.0e-3
training_iters = 30000
display_step = 100

# loading the mnist data
train_data = torchvision.datasets.MNIST(root=data_dir, 
                                        train=True, 
                                        transform=torchvision.transforms.ToTensor(),
                                        download=False
                                        )

# MNIST data's shape is (number,28,28) and value is 0~255
#print(train_data.train_data.size())     # (60000, 28, 28)
# MNIST label is integer and value is 1-10
#print(train_data.train_labels.size())   # (60000)

test_data = torchvision.datasets.MNIST(root=data_dir, 
                                       train = False
                                       )

# shape (2000, 28, 28) value in range(0,1)
test_x = autograd.Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255.0
test_y = test_data.test_labels[:2000]

train_loader = Data.DataLoader(train_data, batch_size, shuffle=False, num_workers=1)

# build prediction graph
print "==> Building a dRNN with %s cells" %cell_type
# x = torch.zeros(batch_size, n_steps, input_dims)
# pred = classification_models.drnn_classification(x, hidden_structs, dilations, n_steps, n_classes, input_dims, cell_type)
model = classification_models.drnn_classification(hidden_structs, dilations, n_classes, input_dims)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()



for iter in range(training_iters):
    
    
    for step, (batch_x, batch_y) in enumerate(train_loader): 
        # (128,1, 28, 28) reshape to (128, 28*28, 1)
        batch_x = autograd.Variable(batch_x.view(-1, n_steps, input_dims))
        batch_y = autograd.Variable(batch_y)
        # reshape inputs
        x_reformat = classification_models._rnn_reformat(batch_x, input_dims, n_steps)
        
        optimizer.zero_grad()
        
        pred = model.forward(x_reformat)
        
        cost = criterion(pred, batch_y)
        
        
        cost.backward()
        optimizer.step()
 
        
        if (step + 1) % display_step == 0:
            print "Iter " + str(iter + 1) + ", Step "+str(step+1)+", Avarage Loss: " + "{:.6f}".format(cost.data[0])

    # validation performance
    x_reformat = classification_models._rnn_reformat(test_x, input_dims, n_steps)
    test_output = model.forward(x_reformat)
    pred_y = torch.max(test_output, 1)[1].data.squeeze()
    accuracy = sum(pred_y == test_y) / float(test_y.size(0))
    print "========> Validation Accuarcy: {:.6f}".format(accuracy) 
         

print "end"


