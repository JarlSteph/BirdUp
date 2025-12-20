import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class BirdPercentModel(nn.Module):
    def __init__(self, in_features=10, hidden_layers=[64, 32, 1]):
        super(BirdPercentModel, self).__init__()
        self.layer_sizes = [in_features] + hidden_layers
        self.stack = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.stack.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))

    def forward(self, x):
        # Iterates through all layers applying ReLU to all but the final layer
        for i in range(len(self.stack) - 1):
            x = F.relu(self.stack[i](x))
        x = torch.sigmoid(self.stack[-1](x))
        return x


def train_model(X_train, y_train, X_val, y_val, model, num_epochs=1000, learning_rate=0.001, val = False):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = X_train
    y_train_tensor = torch.unsqueeze(y_train, -1)
    X_val_tensor = X_val
    y_val_tensor = torch.unsqueeze(y_val, -1)
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if ((epoch+1) % 1000 == 0) and val:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                # Calculate Accuracy (assuming binary classification)
                predictions = (val_outputs > 0.5).float()
                correct = (predictions == y_val_tensor).float().sum()
                accuracy = correct / y_val_tensor.shape[0]
                
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {accuracy:.4f}')
            
    return model

def evaluate(preds: np.ndarray, truths : np.ndarray, cm = False, fOne= False):
    bal_acc = balanced_accuracy_score(truths, preds)
    if fOne:
        f1 = f1_score(truths, preds)
        print(f"F1-Score: {f1:.4f}")

    if cm: 
        cm = confusion_matrix(truths, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix")
        plt.show()
    return bal_acc

def grid_search(X_train, y_train, X_val, y_val, param_grid):
    best_bal_acc = 0
    best_params = {}
    best_model_state = None

    # Iterating through hyperparameters
    for epochs in param_grid['num_epochs']:
        for lr in param_grid['learning_rate']:
            for layers in param_grid['hidden_layers']:
                
                # Initialize model with current layers
                model = BirdPercentModel(in_features=X_train.shape[1], hidden_layers=layers).to(device=device)
                
                # Train model
                trained_model = train_model(X_train, y_train, X_val, y_val, model, num_epochs=epochs, learning_rate=lr, val=False)
                
                # Evaluate model
                trained_model.eval()
                with torch.no_grad():
                    val_outputs = trained_model(X_val)
                    preds = (val_outputs > 0.5).cpu().numpy().astype(int)
                    truths = y_val.cpu().numpy().astype(int)
                    
                    # Using evaluate function to get balanced accuracy
                    current_bal_acc = evaluate(preds, truths)
                
                # Track best hyperparameters
                if current_bal_acc > best_bal_acc:
                    best_bal_acc = current_bal_acc
                    best_params = {'num_epochs': epochs, 'learning_rate': lr, 'hidden_layers': layers}
                    best_model_state = trained_model.state_dict()
                
                print(f"Tested: epochs={epochs}, lr={lr}, layers={layers} | Bal Acc: {current_bal_acc:.4f}")

    print(f"\nBest Balanced Accuracy: {best_bal_acc:.4f}")
    print(f"Best Params: {best_params}")
    
    return best_params, best_model_state

def optim_signmoid_search(X_val, y_val,trained_model):
    sigms = np.linspace(0.0, 0.3, 20)
    best_bal_acc = 0

    with torch.no_grad():
        val_outputs = trained_model(X_val)
        val_outputs_np = val_outputs.cpu().numpy()

        for sigm in sigms:
            preds = (val_outputs_np > sigm).astype(int)
            truths = y_val.cpu().numpy().astype(int)

            current_bal_acc = evaluate(preds, truths)

            if current_bal_acc > best_bal_acc:
                best_bal_acc = current_bal_acc
                best_sigm = sigm

    return best_sigm, best_bal_acc