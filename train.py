import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import get_dataloader
from models.vit_model import create_vit_model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, _ = get_dataloader()
    model = create_vit_model().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99: # 每100个batch打印一次
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), 'vit_cifar10.pth')

if __name__ == "__main__":
    train()