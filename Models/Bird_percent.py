import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm_notebook as tqdm

class BirdPercentModel(nn.Module):
    def __init__(self, in_features=10):
        super(BirdPercentModel, self).__init__()

        self.layer_sizes = [in_features, 64, 32, 1]
        self.stack = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.stack.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))

    def forward(self, x):
        x = F.relu(self.stack[0](x))
        x = F.relu(self.stack[1](x))
        x = torch.sigmoid(self.stack[2](x))
        return x


def train_model(X_train, y_train, model, num_epochs=1000, learning_rate=0.001):

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = X_train
    y_train_tensor = y_train

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model


