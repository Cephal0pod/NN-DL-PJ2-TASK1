import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, trainloader, testloader, epochs=20, lr=0.001, device='cuda', optimizer_type="adam"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Unknown optimizer")

    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()


        avg_loss = running_loss / len(trainloader)
        acc = correct / total
        loss_history.append(avg_loss)
        acc_history.append(acc)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}, Accuracy: {acc*100:.2f}%")


    torch.save(model.state_dict(), f'cifar10_model_{optimizer_type}.pth')
    return loss_history, acc_history
