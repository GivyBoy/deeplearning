import torch

print(torch.cuda.is_available())  # Do we have a GPU? Should return True
print(torch.cuda.device_count())

