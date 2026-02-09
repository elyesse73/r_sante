import torch
import time
import copy
from torchvision import models
from torch import nn
from config import CONFIG

# --- 1. The "Bounded Accuracy" Metric ---
def calc_bounded_accuracy(logits, targets, low_threshold=0.25, high_threshold_on_ambiguity=0.75):
    """
    Calculates accuracy with your strict rules:
    - False if confidence < 25%
    - False if confidence > 75% when experts disagree (ambiguity)
    """
    # Convert logits -> probabilities
    probs = torch.softmax(logits, dim=1)

    # Model prediction (Max confidence and Index)
    confidences, pred_idx = torch.max(probs, dim=1)

    # Expert value for the class chosen by the model
    target_vals = targets.gather(1, pred_idx.view(-1, 1)).squeeze()

    # A. Is it a class validated by at least one expert (>0) ?
    cond_valid = target_vals > 0

    # B. Does the model have at least 25% confidence?
    cond_confident = confidences >= low_threshold

    # C. Respect for ambiguity:
    # If the expert label is pure (>=0.9), 100% confidence is allowed.
    # Otherwise (disagreement), the model MUST NOT exceed 75% confidence.
    is_pure_target = target_vals >= 0.9
    respects_ambiguity = is_pure_target | (confidences <= high_threshold_on_ambiguity)

    # Result: All must be True
    correct_tensor = cond_valid & cond_confident & respects_ambiguity

    return correct_tensor.float().mean().item()

# --- 2. The Training Loop ---
def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, save_path='best_model.pth'):
    since = time.time()

    # Keep a copy of initial weights
    best_model_wts = copy.deepcopy(model.state_dict())

    # CRITICAL CHANGE: Initialize best loss to infinity to minimize it
    # We no longer save based on accuracy because it is unstable with Soft Labels
    best_loss = float('inf')

    # History of losses and accuracies for plotting
    train_loss_history = []
    val_loss_history = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Starting training on {device} for {num_epochs} epochs.")
    print("-" * 50)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Training mode
            else:
                model.eval()   # Evaluation mode

            running_loss = 0.0
            running_accs = [] # List to store Bounded Accuracy scores

            # Loop over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics: Loss
                running_loss += loss.item() * inputs.size(0)

                # --- NEW: Bounded Accuracy Calculation ---
                # We pass raw outputs (logits), the function handles softmax
                batch_acc = calc_bounded_accuracy(outputs, labels, low_threshold=0.25, high_threshold_on_ambiguity=0.75)
                running_accs.append(batch_acc)

            # Scheduler
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            # Calculate epoch averages
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_bounded_acc = sum(running_accs) / len(running_accs) if running_accs else 0.0

            # Store history
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)


            print(f'{phase.upper()} Loss: {epoch_loss:.4f} | Bounded Acc: {epoch_bounded_acc*100:.2f}%')

            # SAVE: Based on LOSS (More mathematically stable)
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
                print(f"    -> New champion (Loss: {best_loss:.4f}) ! Saved as '{save_path}'")

        print() # Newline

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Loss: {best_loss:.4f}')

    # Reload best weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history

def get_convnext_model(num_classes, mode='full'):
    """
    Loads ConvNeXt with 3 training strategies:
    - 'feature_extraction': Freezes everything except the last layer (Fast).
    - 'fine_tune': Freezes the beginning, trains the end (Compromise).
    - 'full': Trains the entire network (Maximum performance).
    """
    print(f"--- Loading {CONFIG['MODEL_TYPE']} model (Mode: {mode}) ---")

    # 1. Load pre-trained weights (ImageNet)
    # Use getattr to dynamically get the model based on CONFIG['MODEL_TYPE']
    model = getattr(models, CONFIG['MODEL_TYPE'])(weights='DEFAULT')

    # 2. Freezing Management
    if mode == 'feature_extraction':
        # Freeze the ENTIRE backbone
        for param in model.features.parameters():
            param.requires_grad = False
        print("-> Backbone FROZEN (only the head will be trained)")

    elif mode == 'fine_tune':
        # Freeze the first blocks, unfreeze the last one (Stage 4 = block '7')
        for name, child in model.features.named_children():
            if name != '7':
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
        print("-> Beginning of backbone FROZEN, Stage 4 UNLOCKED")

    else: # mode == 'full'
        print("-> Entire network is trainable")

    # 3. Replace the classification head
    # ConvNeXt: classifier[2] is the final linear layer
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model