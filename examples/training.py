import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm, trange

def train(model: nn.Module, device, train_loader, loss_fn, optimizer):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

def test_ann(model: nn.Module, device, test_loader, loss_fn):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += loss_fn(output, target).item() * len(target)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)
    return loss / total, correct / total

def train_ann(
    model: nn.Module,
    train_loader,
    test_loader,
    loss_fn,
    epochs=40,
    device=None,
    save_path="tmp/model.pth",
    b_plot_result=False,
    initial_lr=0.01,
) -> nn.Module:
    best_accuracy = 0
    history_loss, history_acc, history_lr = [], [], []

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    pbar = trange(1, epochs+1)
    for epoch in pbar:
        train(model, device, train_loader, loss_fn, optimizer)
        test_loss, test_acc = test_ann(model, device, test_loader, loss_fn)
        scheduler.step()

        history_loss.append(test_loss)
        history_acc.append(test_acc)
        history_lr.append(scheduler.get_last_lr())

        # Save the model if it's the best so far
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), save_path)

        #print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc*100:.4f}%")
        pbar.set_description(f"Loss: {test_loss:.3f}, Accuracy: {best_accuracy*100:.2f}%")
    
    if b_plot_result:
        import matplotlib.pyplot as plt
        # Plot loss and accuracy
        plt.figure(figsize=(8, 8))
        plt.subplot(3, 1, 1)
        plt.plot(range(1, epochs + 1), history_loss, label='Loss')
        plt.title("Training Loss")
        plt.xlabel("Epoch")

        plt.subplot(3, 1, 2)
        plt.plot(range(1, epochs + 1), history_acc, label='Accuracy')
        plt.title("Test Accuracy")
        plt.xlabel("Epoch")

        plt.subplot(3, 1, 3)
        plt.plot(range(1, epochs + 1), history_lr, label='Accuracy')
        plt.title("Learning Rate")
        plt.xlabel("Epoch")

        plt.tight_layout()
        plt.show()

    return model
