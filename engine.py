import torch
from torch.nn import nn


# This file basically contains the blocks for efficiently training the model.

def train_step(model: nn.Module,
               dataloader: torch.utils.data.Dataloader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               max_grad_accumulation_steps: int = 20)-> tuple[float, float]:
    
    N = len(dataloader)

    # assert max_grad_accumulation_steps / N < 0.3, "Let the grad. accumulation steps be less than 30% of total batches."
    
    # set the model to train mode.
    model.train()

    train_loss, train_acc = 0.0, 0.0
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        # retrieving the logits and probs for the input X
        logits, probs = model(X)
        loss = loss_function(logits, y)

        train_loss += loss.item()

        # computing the accuracy as currently performing a classification task
        y_pred_class = torch.argmax(probs)
        train_acc += (y_pred_class == y).sum().item() / len(y)

        optimizer.zero_grad()

        loss.backward()
        # performing updation of weights and zero grad for next backprop
        optimizer.step()

    # Normalizing the train eval metrics.
    train_loss = train_loss / N
    train_acc = train_acc / N

    return train_loss, train_acc