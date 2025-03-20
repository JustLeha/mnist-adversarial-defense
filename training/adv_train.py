import torch
from attack.fgsm import fgsm_attack

def adversarial_train(model, train_loader, criterion, optimizer, epsilon=0.3, epochs=3):
    for epoch in range(epochs):
        for images, labels in train_loader:
            images.requires_grad = True
            adv_images = fgsm_attack(model, images, labels, epsilon, criterion)

            optimizer.zero_grad()
            outputs = model(torch.cat([images, adv_images]))
            loss = criterion(outputs, torch.cat([labels, labels]))
            loss.backward()
            optimizer.step()
        print(f"Adversarial Training Epoch {epoch + 1}, Loss: {loss.item():.4f}")
