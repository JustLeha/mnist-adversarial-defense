import torch
from attack.fgsm import fgsm_attack

def evaluate(model, data_loader, criterion, epsilon=None):
    correct = 0
    total = 0
    for images, labels in data_loader:
        if epsilon is not None:
            images = fgsm_attack(model, images, labels, epsilon, criterion)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return 100 * correct / total
