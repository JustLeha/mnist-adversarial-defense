import torch

def fgsm_attack(model, image, label, epsilon, criterion):
    image.requires_grad = True
    output = model(image)
    loss = criterion(output, label)
    model.zero_grad()
    loss.backward()
    perturbed_image = image + epsilon * image.grad.sign()
    return torch.clamp(perturbed_image, 0, 1)
