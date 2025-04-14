import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 3)
        self.output = nn.Linear(3, 1)

        def forward(self, x):
            x = torch.relu(self.hidden(x))
            x = torch.sigmoid(self.output(x))
            return x

model = SimpleNN()
print(model)
