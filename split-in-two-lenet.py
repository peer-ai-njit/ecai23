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


class Lenet(nn.Module):

  def __init__(self, name, seed):
    super(Lenet, self).__init__()
    self.name = name
    self.seed = seed
    self.net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), 
                                   nn.Sigmoid(),
                                   nn.AvgPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(6, 16, kernel_size=5), 
                                   nn.Sigmoid(),
                                   nn.AvgPool2d(kernel_size=2, stride=2),
                                   nn.Flatten(),
                                   nn.Linear(16 * 5 * 5, 120), 
                                   nn.Sigmoid(),
                                   nn.Linear(120, 84), 
                                   nn.Sigmoid(),
                                   nn.Linear(84, 10))
    self.apply(self.init_weights)

    self.optimizer = torch.optim.SGD(self.parameters(), lr=0.9)
    self.loss_fn = nn.CrossEntropyLoss()

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.to(self.device)

  def forward(self, x):
    y = x.view(-1, 1, 28, 28).to(self.device)
    y = self.net(y)
    return y

  def init_weights(self, m):
      if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.manual_seed(self.seed)
        nn.init.xavier_uniform_(m.weight)

  def learn(self, x, y):
      x = x.to(self.device)
      y = y.to(self.device)
      self.train()
      yhat = self.forward(x)
      loss = self.loss_fn(yhat, y)
      loss.backward()
      self.optimizer.step()
      self.optimizer.zero_grad()
      return loss

      
  def ensembled_forward(self, X):
      soft_max = nn.Softmax(dim = 0)
      self.eval()
      X = X.to(self.device)
      preds = self.forward(X)
      prob_preds = soft_max(preds)
      log_prob = torch.log(prob_preds)
      return log_prob

  def accuracy(self, test_iter):
      n_samples = 0 
      n_correct = 0
      self.eval()
      for batch in test_iter:
        batch[0] = batch[0].to(self.device)
        batch[1] = batch[1].to(self.device)
        trues = batch[1]
        preds = self.forward(batch[0]).argmax(axis=1)
        n_samples = n_samples + batch[1].shape[0]
        n_correct = n_correct + (trues==preds).sum()
        #break
      accuracy = np.true_divide(n_correct.cpu().numpy(), n_samples)
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
    temp_model = Lenet(f"{i+1}", i+1)
    for epoch in range (i+1):
      for batch in train_iter:
        temp_model.learn(batch[0], batch[1])  
    models = np.append(models, temp_model)
  models = np.append(models, Model_10())
  return models

def initialize_models_no_pretrain():
  models = np.array([])
  for i in range(9):
    temp_model = Lenet(f"{i+1}", i+1)
    models = np.append(models, temp_model)
  models = np.append(models, Model_10())
  return models

def split_in_two_training(models, epochs, train_iter, valid_iter):

  accuracy = 0
  for model in models:
    if model.accuracy(valid_iter) > accuracy:
      mentor = model
      accuracy = model.accuracy(valid_iter)
  for model in models:
    if model.name != mentor.name:
      for epoch in range(epochs):
        if mentor.name == "10":
          for batch in train_iter:
            model.learn(batch[0], batch[1])
        else:
          for batch in train_iter:
            model.learn(batch[0], mentor.forward(batch[0]).argmax(axis=1))

def policy_btb(models, valid_iter):
  test_accuracies = np.array([])
  for model in models:
    test_accuracies = np.append(test_accuracies, model.accuracy(valid_iter))
  models = models[test_accuracies.argsort()]
  group_1 = []
  group_2 = []
  for i in range(10):
    if i < 4 or i == 8:
      group_2 = np.append(group_2, models[i])
    elif i > 3 and i < 8 or i == 9:
      group_1 = np.append(group_1, models[i])
  return group_1, group_2

def policy_eq(models, valid_iter):
  test_accuracies = np.array([])
  for model in models:
    test_accuracies = np.append(test_accuracies, model.accuracy(valid_iter))
  models = models[test_accuracies.argsort()]
  group_1 = []
  group_2 = []
  for i in range(10):
    if i < 8 and i % 2 == 0 or i == 8:
      group_2 = np.append(group_2, models[i])
    elif i < 8 and (i % 2) != 0 or i == 9:
      group_1 = np.append(group_1, models[i])
  return group_1, group_2

def policy_rgbt(models):
  np.random.shuffle(models)
  group_1 = []
  group_2 = []
  i = 0
  for model in models:
    if i < 5 :
      group_1.append(model)
    else :
      group_2.append(model)
    i += 1
  return group_1, group_2

def policy_oo(models):
  np.random.shuffle(models)
  group_1 = []
  for model in models:
    if model.name == "10":
      group_1 = np.append(group_1, model)
      break
  i = 0
  for model in models:
    if model.name != "10":
      group_1 = np.append(group_1, model)
      i += 1
      if i == 4:
        break
  return group_1


train = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))                                          
train_set = torch.utils.data.DataLoader(train, 
                                           batch_size=200)
test_set = torch.utils.data.DataLoader(test,
                                          batch_size=200)
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    #Ensembled and Average Accuracy Calculations For Star Mode and Policy BTB
    models_lst = initialize_models_pretrain()
    avg_test_accuracy_history_btb = []
    ensmbl_test_accuracy_history_btb = []
    for t in range(n_rounds):
      group_1, group_2 = policy_btb(models_lst)
      split_in_two_training(group_1, pair_epochs, train_iter, valid_iter)
      split_in_two_training(group_2, pair_epochs, train_iter, valid_iter)
    
      test_accuracies = []
      temp_ensmbl_acc = []
    
      for model in models_lst:
        if model.name != "10":
          test_accuracies.append(model.accuracy(test_iter))
      avg_test_accuracy_history_btb.append(np.average(test_accuracies))
    
      for batch in test_iter:
        log_probs = torch.zeros(batch_size, 10).to("cuda")
        for model in models_lst:
          if model.name != "10":
            log_probs += model.ensembled_forward(batch[0])
        temp_ensmbl_acc.append(float(ensembled_accuracy(batch[0], batch[1], log_probs)))
      ensmbl_test_accuracy_history_btb.append(np.average(temp_ensmbl_acc))  
    
    #Ensembled and Average Accuracy Calculations For Star Mode and Policy EQ
    models_lst = initialize_models_pretrain()
    avg_test_accuracy_history_eq = []
    ensmbl_test_accuracy_history_eq = []
    for t in range(n_rounds):
      group_1, group_2 = policy_eq(models_lst)
      split_in_two_training(group_1, pair_epochs, train_iter, valid_iter)
      split_in_two_training(group_2, pair_epochs, train_iter, valid_iter)
    
      test_accuracies = []
      temp_ensmbl_acc = []
    
      for model in models_lst:
        if model.name != "10":
          test_accuracies.append(model.accuracy(test_iter))
      avg_test_accuracy_history_eq.append(np.average(test_accuracies))
    
      for batch in test_iter:
        log_probs = torch.zeros(batch_size, 10).to("cuda")
        for model in models_lst:
          if model.name != "10":
            log_probs += model.ensembled_forward(batch[0])
        temp_ensmbl_acc.append(float(ensembled_accuracy(batch[0], batch[1], log_probs)))
      ensmbl_test_accuracy_history_eq.append(np.average(temp_ensmbl_acc))  
      
    #Ensembled and Average Accuracy Calculations For Star Mode and Policy RGBT
    models_lst = initialize_models_pretrain()
    avg_test_accuracy_history_rgbt = []
    ensmbl_test_accuracy_history_rgbt = []
    for t in range(n_rounds):
      group_1, group_2 = policy_rgbt(models_lst)
      split_in_two_training(group_1, pair_epochs, train_iter, valid_iter)
      split_in_two_training(group_2, pair_epochs, train_iter, valid_iter)
    
      test_accuracies = []
      temp_ensmbl_acc = []
    
      for model in models_lst:
        if model.name != "10":
          test_accuracies.append(model.accuracy(test_iter))
      avg_test_accuracy_history_rgbt.append(np.average(test_accuracies))
    
      for batch in test_iter:
        log_probs = torch.zeros(batch_size, 10).to("cuda")
        for model in models_lst:
          if model.name != "10":
            log_probs += model.ensembled_forward(batch[0])
        temp_ensmbl_acc.append(float(ensembled_accuracy(batch[0], batch[1], log_probs)))
      ensmbl_test_accuracy_history_rgbt.append(np.average(temp_ensmbl_acc)) 
      
    #Ensembled and Average Accuracy Calculations For Star Mode and Policy OO
    models_lst = initialize_models_pretrain()
    avg_test_accuracy_history_oo = []
    ensmbl_test_accuracy_history_oo = []
    for t in range(n_rounds):
      group_1 = policy_oo(models_lst)
      split_in_two_training(group_1, pair_epochs, train_iter, valid_iter)
    
      test_accuracies = []
      temp_ensmbl_acc = []
    
      for model in models_lst:
        if model.name != "10":
          test_accuracies.append(model.accuracy(test_iter))
      avg_test_accuracy_history_oo.append(np.average(test_accuracies))
    
      for batch in test_iter:
        log_probs = torch.zeros(batch_size, 10).to("cuda")
        for model in models_lst:
          if model.name != "10":
            log_probs += model.ensembled_forward(batch[0])
        temp_ensmbl_acc.append(float(ensembled_accuracy(batch[0], batch[1], log_probs)))
      ensmbl_test_accuracy_history_oo.append(np.average(temp_ensmbl_acc))  
      
if __name__ == '__main__':
    main()