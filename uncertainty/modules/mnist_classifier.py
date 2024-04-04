import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import torchvision

from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os
import numpy as np
    
def train_classifier(x, targets, classifier, optimizer, criterion, device):
    classifier.zero_grad()
    x = x.to(device)
    targets = targets.to(device)
    logits = classifier(x)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    accuracy = torch.sum(torch.argmax(logits, dim=1) == targets)
    return loss.item(), accuracy.item()


def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (x, targets) in enumerate(data_loader):
            
        x = x.to(device)
        targets = targets.to(device)

        logits = model(x)
        _, predicted_labels = torch.max(logits, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
    
def eval_model(data_loader, classifier, criterion, device):
    total_loss = 0.0
    correct_pred = 0
    total_samples = 0
    
    for batch_idx, (x, targets) in enumerate(data_loader):
        x, targets = x.to(device), targets.to(device)
        logits = classifier(x)
        loss = criterion(logits, targets)

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        # Compute accuracy
        _, predicted_labels = torch.max(logits, 1)
        correct_pred += (predicted_labels == targets).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct_pred / total_samples * 100

    return avg_loss, accuracy

def get_preds(model, data_loader, device):
    predicted_labels_list = []
    targets_list = []
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():  # We don't need gradients for inference
        for i, (x, targets) in enumerate(data_loader):
            x = x.to(device)
            targets = targets.to(device)

            logits = model(x)
            _, predicted_labels = torch.max(logits, 1)
            predicted_labels_list += ([int(el) for el in list(predicted_labels)])
            targets_list += ([int(el) for el in list(targets)])
    
    return predicted_labels_list, targets_list
from sklearn.metrics import confusion_matrix
import seaborn as sns

def calculate_confusion_matrix(model, data_loader, device, epoch = None, save_path = None, name = None):
    y_pred, y_true = get_preds(model, data_loader, device)
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=list(range(10)),
            yticklabels=list(range(10)))
    plt.ylabel('Actual', fontsize=13)
    plt.xlabel('Prediction', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    

    if save_path and name:
        # Save the figure in the specified folder
        save_filename = os.path.join(save_path, name, f'cm_epoch_{epoch}.png')
        # save_filename = os.path.join(save_path, name, 'generated_plots.png')
        plt.savefig(save_filename, dpi=300)
    plt.close()
    
    
    

def train_epoch(train_loader, test_loader, classifier, optimizer, criterion, device, scheduler = None):
    total_loss = 0.0
    total_samples = 0
    correct_pred = 0
    classifier.train()
    
    for batch_idx, (x, targets) in enumerate(train_loader):
        batch_size = x.size(0)
        x, targets = x.to(device), targets.to(device)
        
        # # Train classifier
        # classifier_loss = train_classifier(x, classifier, optimizer, criterion, device)
        # Train classifier
        classifier_loss, correct = train_classifier(x, targets, classifier,
                                                    optimizer, criterion, device)

        # Update total losses
        total_loss += classifier_loss * batch_size
        total_samples += batch_size
        correct_pred += correct
        
    if scheduler:
        scheduler.step()
    # Calculate average loss over all samples
    avg_loss_train = total_loss / total_samples
    # Calculate accuracy
    accuracy_train = correct_pred / total_samples * 100
    
    classifier.eval()
    with torch.no_grad():
        avg_loss_test, accuracy_test = eval_model(test_loader, classifier, criterion, device)
        

    return avg_loss_train, accuracy_train, avg_loss_test, accuracy_test


def train(num_epochs,                  # Number of training epochs
          train_loader,                 # DataLoader providing training data
          test_loader,#test loader
          classifier,                  # classifier
          optimizer,                   # Optimizers for classifier
          criterion,                   # Loss function criterion
          device,                      # Device to perform training on (e.g., 'cuda' or 'cpu')
          plot_process=False,          # Whether to plot the training process
          save_path=None,              # Path to save the plots (if plotting is enabled)
          name="generated_plots.png",  # Name of the saved plot file, use the main features in name
          info_n = 20,                 #write info(metrics, vars and etc.) every info_n epoch
          scheduler=None,            # Scheduler for classifier optimizer (optional)
          save_model_name = None     #name of model to save it or None to not save it
         ):
    """
    Returns:
    - D_losses_final (list): List of Discriminator losses for each epoch.
    - G_losses_final (list): List of Generator losses for each epoch.
    - Variances (list): List of variances during training (if applicable).
    """
    classifier_losses__train_final = []
    classifier_losses__test_final = []
    classifier_acc__train_final = []
    classifier_acc__test_final = []
    
    
    if save_path:
        create_folder(save_path, name)

    for epoch in tqdm(range(num_epochs)):
    
        avg_loss_train, accuracy_train, avg_loss_test, accuracy_test = train_epoch(train_loader, test_loader, classifier, optimizer, criterion, device, scheduler)
        
        classifier_losses__train_final.append(avg_loss_train)
        classifier_losses__test_final.append(avg_loss_test)
        
        
        classifier_acc__train_final.append(accuracy_train)
        classifier_acc__test_final.append(accuracy_test)
        if epoch % info_n == 0: 
            print(f"epoch [{epoch}/{num_epochs}], average classifier_loss: train -- {avg_loss_train:.4f}, test -- {avg_loss_test:.4f} , average classifier_accuracy: train -- {accuracy_train:.4f}, test -- {accuracy_test:.4f}")
            
        
    if plot_process:
        process_name = f'process__losses_train_{avg_loss_train:.4f}_test_{avg_loss_test:.4f}__accuracy_train_{accuracy_train:.4f}_test_{accuracy_test:.4f}'
        plot_training_progress(classifier_losses__train_final, classifier_losses__test_final,
                               classifier_acc__train_final, classifier_acc__test_final,
                               save_path = save_path, name = name,
                               process_name = process_name);
        
    if save_model_name:
        save_model(classifier, save_path = save_path, name = name, name2 = save_model_name)

            
    return
    
def create_folder(base_path, folder_name):  
    full_path = os.path.join(base_path, folder_name)
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Folder '{folder_name}' created at '{base_path}'.")
    else:
        pass
    
def plot_training_progress(train_losses, test_losses,
                                  train_acc, test_acc,
                                  save_path, name, process_name = 'process'):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.title('Losses')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Test')
    plt.title("Accuracy")
    plt.legend()

    if save_path:
        save_filename = os.path.join(save_path, name, f"{process_name}.png")
        plt.savefig(save_filename, dpi=300)

    plt.tight_layout()
    plt.close() 
    
def save_model(model, save_path, name, name2):
    filepath = os.path.join(save_path, name, name2)
    model.eval()
    torch.save(model.state_dict(), filepath)
    print(f"Model saved at: {filepath}")
    
    

    
    
