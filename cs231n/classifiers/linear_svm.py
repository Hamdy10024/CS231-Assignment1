import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).
  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        
        dW[:, j] += X[i]  
        dW[:, y[i]] -= X[i]
       
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) 

  scores =np.dot(X,W) #n*c
  correct_scores = scores[range(len(y)), y] 
  margin = scores.T - correct_scores.T +1
  margin = margin.T
  margin =np.maximum(margin,0);
  margin[range(len(y)), y]=0
  loss = np.sum(margin)/len(y)
  factor = np.ones((len(y),10)) #n*10
  factor[range(len(y)), y] = -1
  dW = np.dot(X.T,factor)/len(y) 
  print(dW.shape)
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  return loss, dW
x = np.random.randint(low=1, high=100, size=(10,2))
y = np.random.randint(low = 0,high = 9,size = 10).T
W = np.random.uniform(low=0.0, high=0.1, size=(2,10))
svm_loss_naive(W,x,y,0.1)

 
