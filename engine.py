import torch
import contextlib
from torch.nn import nn


# This file basically contains the blocks for efficiently training the model.

def step(model: nn.Module,
               dataloader: torch.utils.data.Dataloader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               type_step: str)-> tuple[float, float]:
    
    N = len(dataloader)
    # assert max_grad_accumulation_steps / N < 0.3, "Let the grad. accumulation steps be less than 30% of total batches."
    # set the model to train mode.

    step_loss, step_acc = 0.0, 0.0

    model = model.train() if type_step == 'train' else model.eval()
    
    with contextlib.nullcontext() if type_step == 'train' else torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # retrieving the logits and probs for the input X
            logits, probs = model(X)
            loss = loss_function(logits, y)

            step_loss += loss.item()

            # computing the accuracy as currently performing a classification task
            y_pred_class = torch.argmax(probs)
            step_acc += (y_pred_class == y).sum().item() / len(y)

            if type_step == 'train':
                # performing updation of weights and zero grad for next backprop
                optimizer.zero_grad()
                loss.backward()
                # Updating the weights with calculated grads dw = dw - lr * dr
                optimizer.step()

        # Normalizing the train eval metrics.
        step_loss = step_loss / N
        step_acc = step_acc / N

    return step_loss, step_acc
