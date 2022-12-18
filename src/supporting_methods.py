import numpy as np

def ReLu(x):
  return x * (x >= 0)

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x))