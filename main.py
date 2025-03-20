import torch
import torch.optim as optim
import torch.nn as nn
from model.cnn import CNN
from data.dataset import get_data_loaders
from training.train import train_model
from training.adv_train import adversarial_train
from evaluation.evaluate import evaluate

# Инициализация
train_loader, test_loader = get_data_loaders()
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Обучение
train_model(model, train_loader, criterion, optimizer)
adversarial_train(model, train_loader, criterion, optimizer, epsilon=0.3)

# Оценка
accuracy_before = evaluate(model, test_loader, criterion)
accuracy_after_attack = evaluate(model, test_loader, criterion, epsilon=0.3)

print(f"Точность до атаки: {accuracy_before:.2f}%")
print(f"Точность после атаки: {accuracy_after_attack:.2f}%")
