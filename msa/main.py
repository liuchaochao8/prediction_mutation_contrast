
import numpy as np
import torch
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
from EsmModel import get_model, EsmDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_set(train_file_path, test_file_path):
    # am_class = {'ambiguous': 0, 'pathogenic': 1, 'benign': 2}
    train_dataset = EsmDataset(
        file_path=train_file_path,
        file_type='csv'
        # am_class=am_class
    )
    test_dataset = EsmDataset(
        file_path=test_file_path,
        file_type='csv'
        # am_class=am_class
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,  # Adjust based on your GPU memory
        shuffle=False,  # Set to True if needed
        num_workers=0,  # 0 to avoid CUDA issues
        pin_memory=True,  # Optional: speeds up data transfer to GPU
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,  # Adjust based on your GPU memory
        shuffle=False,  # Set to True if needed
        num_workers=0,  # 0 to avoid CUDA issues
        pin_memory=True,  # Optional: speeds up data transfer to GPU
    )
    return train_dataloader, test_dataloader


def fetch_data(batch):
    variant_sequence = batch['variant_sequence']
    labels_1 = batch['am_pathogenicity'].to(device)
    return variant_sequence, labels_1


def compute_accuracy(predictions, labels):
    # Get the predicted class (max value index)
    # _, predicted_classes = torch.max(predictions, 1)
    # Compute the number of correct predictions
    # correct = (predicted_classes == labels).sum().item()

    # max_value, max_index = torch.max(predictions,1).values()
    accuracy = 0
    # accuracy = (torch.sum(predictions == labels)).item() / labels.shape[0]
    # accuracy = (sum((predictions == labels).tolist()) / labels.shape[0])
    # accuracy = correct / labels.size(0)
    return accuracy


def compute_loss(predictions, labels):
    return CrossEntropyLoss()(torch.tensor(predictions).type(torch.float), torch.tensor(labels).type(torch.long))


def train(model, train_dataloader, val_dataloader, optimizer, scheduler, n_epochs=20):
    best_val_loss = np.inf
    f = open("loss.txt", "w")
    for epoch in trange(n_epochs):
        lr = scheduler.optimizer.param_groups[0]["lr"]
        model.train()
        avg_train_loss = 0
        avg_task_1_loss = 0
        avg_task_1_acc = 0
        for batch in train_dataloader:
            variant_sequence, labels_1 = fetch_data(batch)
            optimizer.zero_grad()
            task_1_output = model(variant_sequence) # Forward pass through the full model
            task_1_loss = compute_loss(task_1_output, labels_1)  # Compute losses for each task (considering class weights)
            # print(task_1_output.size(), labels_1.size())
            # print(task_1_output, labels_1)
            task_1_acc = compute_accuracy(task_1_output, labels_1)  # Compute accuracies for each task

            task_1_loss.requires_grad_(True)
            task_1_loss.backward()
            optimizer.step()
            avg_task_1_loss += task_1_loss.item()
            avg_task_1_acc += task_1_acc
        # Average losses and accuracies for this epoch
        avg_train_loss /= len(train_dataloader)
        avg_task_1_loss /= len(train_dataloader)
        avg_task_1_acc /= len(train_dataloader)

        print(f"Epoch {epoch+1}/{n_epochs}, LR: {lr:.5f}, "f"Train Loss: {avg_train_loss:.4f}, "
              f"Task 1 Loss: {avg_task_1_loss:.4f}, "f"Task 1 Accuracy: {avg_task_1_acc:.4f}")
        f.write(str(avg_task_1_loss) + '\n')

        task_1_val_loss, task_1_val_acc = validate(model, val_dataloader) # Validation phase
        scheduler.step(task_1_val_loss)  # Update the learning rate based on validation loss
        if task_1_val_loss < best_val_loss:
            best_val_loss = task_1_val_loss
            print(f"New best validation loss: {task_1_val_loss:.4f}, saving model...")
            torch.save(model.state_dict(), 'best_model_MSA_ProMEP.pth')

    f.close()


def validate(model, val_dataloader):
    model.eval()
    avg_task_1_loss = 0
    avg_task_1_acc = 0
    with torch.no_grad():
        for batch in val_dataloader:
            variant_sequence, labels_1 = fetch_data(batch)
            task_1_output = model(variant_sequence)  # Forward pass through the model
            task_1_loss = compute_loss(task_1_output, labels_1) # Compute the task-specific losses
            task_1_acc = compute_accuracy(task_1_output, labels_1)  # Compute accuracies for each task
            avg_task_1_loss += task_1_loss.item()
            avg_task_1_acc += task_1_acc
    # Average losses and accuracies for validation set
    avg_task_1_loss /= len(val_dataloader)
    avg_task_1_acc /= len(val_dataloader)
    print(f"Validation Task 1 Loss: {avg_task_1_loss:.4f}, " f"Task 1 Accuracy: {avg_task_1_acc:.4f}")
    return avg_task_1_loss, avg_task_1_acc


def main():

    # train_file_path = "train_ProMEP5.csv"
    # test_file_path = "train_ProMEP5.csv"
    train_file_path = "../data/little_muta_protein_300.csv"
    test_file_path = "../data/little_muta_protein_300.csv"
    train_dataloader, test_dataloader = data_set(train_file_path, test_file_path)
    optimizer = optim.Adam(get_model(device).parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=5, min_lr=1e-5)
    n_epochs = 100
    train(get_model(device), train_dataloader, test_dataloader, optimizer, scheduler, n_epochs)



if __name__ == "__main__":
    main()