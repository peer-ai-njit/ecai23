import torch
import torch.nn as nn
import numpy as np
import random
from random import randint
import torchvision
import torch.nn.functional as F
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms


class ResNet(nn.Module):
  def __init__(self, name, in_channels):
    super(ResNet, self).__init__()
    self.name = name
    self.inchannels = in_channels

    # Load a pretrained resnet model from torchvision.models in Pytorch
    self.model = models.resnet18(pretrained=True)

    # Change the input layer to take Grayscale image, instead of RGB images.
    # Hence in_channels is set as 1 or 3 respectively
    # original definition of the first layer on the ResNet class
    # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Change the output layer to output 10 classes instead of 1000 classes
    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Linear(num_ftrs, 10)

    '''****************************************************************************************************************************************************'''
    self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
    self.loss_fn = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.to(self.device)
    '''****************************************************************************************************************************************************'''
  def reset_parameter(self):
    self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.name = self.name
    self.inchannels = self.in_channels
    self.model = models.resnet18(pretrained=True)
        # Change the output layer to output 10 classes instead of 1000 classes
    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Linear(num_ftrs, 10)

    '''****************************************************************************************************************************************************'''
    self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
    self.loss_fn = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.to(self.device)
    '''****************************************************************************************************************************************************'''


  def forward(self, x):
    y = self.model(x)
    return y

  def learn(self, x, y):
    x = x.to(self.device)
    y = y.to(self.device)
    self.train()
    yhat = self.forward(x)
    loss = self.loss_fn(yhat, y)
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

  def ensembled_forward(self, x):
    soft_max = nn.Softmax(dim = 0)
    self.eval()
    x = x.to(self.device)
    preds = self.forward(x)
    prob_preds = soft_max(preds)
    log_prob = torch.log(prob_preds)
    return log_prob

  def accuracy(self, test_iter):
    accuracies = []
    n_samples = 0
    n_correct = 0
    self.eval()
    for data in test_iter:
      x = data[0].to(self.device)
      y = data[1].to(self.device)
      yhat = self.forward(x).argmax(axis = 1)
      accuracies.append(accuracy_score(y.cpu(), yhat.cpu()))
    accuracy = np.average(accuracies)
    return accuracy

def ensembled_accuracy(y, log_probs):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  y = y.to(device)
  trues = y
  preds = log_probs.argmax(axis=1)
  n_samples = y.shape[0]
  n_correct = (trues==preds).sum()
  accuracy = np.true_divide(n_correct.cpu().numpy(), n_samples)  #GE modified
  return accuracy

class Model_10():
  def __init__(self):
    self.name = "10"
  def accuracy(self, test_iter):
    return 1


def initialize_models_pretrain():
  models = np.array([])
  for i in range(9):
    temp_model = ResNet(f"{i+1}", 1)
    for epoch in range (i+1):
      for batch in train_iter:
        temp_model.learn(batch[0], batch[1])
    models = np.append(models, temp_model)
  models = np.append(models, Model_10())
  return models

def initialize_models_no_pretrain():
  models = np.array([])
  for i in range(9):
    temp_model = ResNet(f"{i+1}", 1)
    models = np.append(models, temp_model)
  models = np.append(models, Model_10())
  return models

def split_in_five_training(pair, epochs, train_iter, valid_iter):
  if pair[0].accuracy(valid_iter) > pair[1].accuracy(valid_iter):
    mentor = pair[0]
    mentee = pair[1]
  else:
    mentor = pair[1]
    mentee = pair[0]

  for epoch in range(epochs):
    for batch in train_iter:
      # Model 10 will always be the mentor because its test accuracy is always 1
      if mentor.name == "10":
        mentee.learn(batch[0], batch[1])
      else:
        mentee.learn(batch[0], mentor.forward(batch[0]).argmax(axis=1))


def policy_btb(models, valid_iter):
  # Models sorted from smallest test accuracy to largest, then paired accordingly.
  test_accuracies = np.array([])
  for model in models:
    test_accuracies = np.append(test_accuracies, model.accuracy(valid_iter))
  models = models[test_accuracies.argsort()]

  # After models list is sorted, pairs are created per the project instructions.
  pairs = []
  for model_1, model_2 in zip(models[:5], models[5:]):
    pairs.append((model_1, model_2))
  return pairs

def policy_eq(models, valid_iter):
  test_accuracies = np.array([])
  for model in models:
    test_accuracies = np.append(test_accuracies, model.accuracy(valid_iter))
  models = models[test_accuracies.argsort()]
  pairs = []
  for model_1, model_2 in zip(models[:5], models[:4:-1]):
    pairs.append((model_1, model_2))
  return pairs

def policy_rgbt(models):
  np.random.shuffle(models)
  pairs = []
  for model_1, model_2 in zip(models[:5], models[5:]):
    pairs.append((model_1, model_2))
  return pairs

def policy_oo(models):
  np.random.shuffle(models)
  pairs = []
  for model in models:
    if model.name == "10":
      pairs.append(model)
      break
  for model in models:
    if model.name != "10":
      pairs.append(model)
      break
  return pairs


train = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))
train_set = torch.utils.data.DataLoader(train,
                                           batch_size=200)
test_set = torch.utils.data.DataLoader(test,
                                          batch_size=200)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iter = []
valid_iter = []
test_iter = []
for x,y in train_set:
  train_iter.append([x,y])
for x,y in test_set:
  test_iter.append([x,y])
for j in range(50):
  valid_iter.append(train_iter.pop(randint(0, len(train_iter)-1)))

batch_size = 200
n_rounds = 10
pair_epochs = 1

def main():
    #Ensembled and Average Accuracy Calculations For Policy BTB
    models_lst = initialize_models_pretrain()
    avg_test_accuracy_history_btb = []
    ensmbl_test_accuracy_history_btb = []
    for t in range(n_rounds):
      pairs = policy_btb(models_lst)
      for pair in pairs:
        split_in_five_training(pair, pair_epochs, train_iter, valid_iter)
      test_accuracies = []
      temp_ensmbl_acc = []

      for model in models_lst:
        if model.name != "10":
          test_accuracies.append(model.accuracy(test_iter))
      avg_test_accuracy_history_btb.append(np.average(test_accuracies))

      for batch in test_iter:
        log_probs = torch.zeros(batch_size,10).to("cuda")
        for model in models_lst:
          if model.name != "10":
            log_probs += model.ensembled_forward(batch[0])
        temp_ensmbl_acc.append(float(ensembled_accuracy(batch[0], batch[1], log_probs)))
      ensmbl_test_accuracy_history_btb.append(np.average(temp_ensmbl_acc))

    #Ensembled and Average Accuracy Calculations For Policy EQ
    models_lst = initialize_models_pretrain()
    avg_test_accuracy_history_eq = []
    ensmbl_test_accuracy_history_eq = []
    for t in range(n_rounds):
      pairs = policy_eq(models_lst)
      for pair in pairs:
        split_in_five_training(pair, pair_epochs, train_iter, valid_iter)
      test_accuracies = []
      temp_ensmbl_acc = []

      for model in models_lst:
        if model.name != "10":
          test_accuracies.append(model.accuracy(test_iter))
      avg_test_accuracy_history_eq.append(np.average(test_accuracies))

      for batch in test_iter:
        log_probs = torch.zeros(batch_size,10).to("cuda")
        for model in models_lst:
          if model.name != "10":
            log_probs += model.ensembled_forward(batch[0])
        temp_ensmbl_acc.append(float(ensembled_accuracy(batch[0], batch[1], log_probs)))
      ensmbl_test_accuracy_history_eq.append(np.average(temp_ensmbl_acc))


    #Ensembled and Average Accuracy Calculations For Policy RGBT
    models_lst = initialize_models_pretrain()
    avg_test_accuracy_history_rgbt = []
    ensmbl_test_accuracy_history_rgbt = []
    for t in range(n_rounds):
      pairs = policy_rgbt(models_lst)
      for pair in pairs:
        split_in_five_training(pair, pair_epochs, train_iter, valid_iter)
      test_accuracies = []
      temp_ensmbl_acc = []

      for model in models_lst:
        if model.name != "10":
          test_accuracies.append(model.accuracy(test_iter))
      avg_test_accuracy_history_rgbt.append(np.average(test_accuracies))

      for batch in test_iter:
        log_probs = torch.zeros(batch_size,10).to("cuda")
        for model in models_lst:
          if model.name != "10":
            log_probs += model.ensembled_forward(batch[0])
        temp_ensmbl_acc.append(float(ensembled_accuracy(batch[0], batch[1], log_probs)))
      ensmbl_test_accuracy_history_rgbt.append(np.average(temp_ensmbl_acc))


    #Ensembled and Average Accuracy Calculations For Policy OO
    models_lst = initialize_models_pretrain()
    avg_test_accuracy_history_oo = []
    ensmbl_test_accuracy_history_oo = []
    for t in range(n_rounds):
      pairs = policy_oo(models_lst)
      split_in_five_training(pairs, pair_epochs, train_iter, valid_iter)
      test_accuracies = []
      temp_ensmbl_acc = []

      for model in models_lst:
        if model.name != "10":
          test_accuracies.append(model.accuracy(test_iter))
      avg_test_accuracy_history_oo.append(np.average(test_accuracies))

      for batch in test_iter:
        log_probs = torch.zeros(batch_size,10).to("cuda")
        for model in models_lst:
          if model.name != "10":
            log_probs += model.ensembled_forward(batch[0])
        temp_ensmbl_acc.append(float(ensembled_accuracy(batch[0], batch[1], log_probs)))
      ensmbl_test_accuracy_history_oo.append(np.average(temp_ensmbl_acc))


if __name__ == '__main__':
    main()
