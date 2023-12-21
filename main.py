import torch
import torch.nn as nn

print(torch.cuda.is_available())  # Do we have a GPU? Should return True
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())

torch.manual_seed(17)  # computers a (pseudo) random, so specifying a seed allows for reproducibility

""" 
You've probably used the conv layers that pytorch provides, but do you know how they are initialized and what values 
are used? The following code gives an idea of how the weights are initialized (randomly, I presume).
"""
conv1d = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, stride=1)
print(conv1d.weight)

conv2d = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(3, 3), stride=(1, 1))
print(conv2d.weight)

# pip freeze > requirements.txt # saves all the packages in the current environment to a file called requirements.txt
# use pip install -r requirements.txt to install all the packages in the file
