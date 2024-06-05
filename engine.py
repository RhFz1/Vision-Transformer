import os
import torch
import contextlib
from torch.nn import nn
from utils import save_model
from dotenv import load_dotenv
from typing import Dict, List, Tuple

load_dotenv()

# This file basically contains the blocks for efficiently training the model.

def step(model: nn.Module,
         dataloader: torch.utils.data.Dataloader,
         loss_function: torch.nn.Module,
         optimizer: torch.optim.Optimizer,
         device: torch.device,
         type_step: str)-> Tuple[float, float]:
    
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
            y_pred_class = torch.argmax(probs, dim = 1)
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

def train(model: nn.Module,
          train_dataloader: torch.utils.data.Dataloader,
          validation_dataloader: torch.utils.data.Dataloader,
          test_dataloader: torch.utils.data.Dataloader,
          optimizer: torch.optim.Optimizer,
          loss_function: torch.nn.Module,
          epochs: int,
          eval_iters: int, 
          eval_interval: int,
          model_name: str,
          device: torch.device
          )-> Dict[str, list]:
    # Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):

        # Train the model initially
        train_loss, train_acc = step(model=model,
                                     dataloader=train_dataloader,
                                     loss_function=loss_function,
                                     optimizer=optimizer,
                                     device=device,
                                     type_step='train')
        
        result_string = f"Epoch {epoch + 1}: train_loss: {train_loss: .4f}, train_acc: {train_acc: .4f}"
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        
        if (epoch != 0 and epoch % eval_interval == 0):

            k = results['val_loss']

            # Accumulating the val_loss for eval_iters for smooth tracking
            val_loss, val_acc = 0.0, 0.0
            for itr in range(eval_iters):
                cur_val_loss, cur_val_acc = step(model=model,
                                                 dataloader=validation_dataloader,
                                                 loss_function=loss_function,
                                                 optimizer=optimizer,
                                                 device=device,
                                                 type_step='val')
                val_loss += cur_val_loss
                val_acc += cur_val_loss

            # Normalizing the val_loss
            val_loss = val_loss / eval_iters
            val_acc = val_acc / eval_iters


            # Evaluating with previous scores to save the model
            if len(k):
                if k[-1] < val_loss:
                    save_model(
                        model=model,
                        optimizer=optimizer,
                        target_dir=os.environ.get('model_registry'),
                        model_name=model_name
                    )

            # Appending to results, mind it val_loss array shall be lesser in size than train_loss
            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)

            result_string += "val_loss: {val_loss: .4f}, val_acc: {val_acc: .4f}"

        # Printing epoch wise results.
        print(result_string)

    # Computing the test loss once the training is complete.
    test_loss, test_acc = step(model=model,
                               dataloader=test_dataloader,
                               loss_function=loss_function,
                               optimizer=optimizer,
                               device=device,
                               type_step='test')
    
    results['train_loss'].append(test_loss)
    results['test_acc'].append(test_acc)

    print("Training Complete!!")
    print(f"After training for {epochs} achieved test_loss: {test_loss}, test_acc: {test_acc}")

    return results