from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]

    for i in range(N):
      scores = X[i].dot(W)
      # numerical stability
      scores -= np.max(scores)
      # softmax
      softmax = np.exp(scores)/np.sum(np.exp(scores))
      # cross-entropy
      loss += -np.log(softmax[y[i]])
      dscores = softmax
      dscores[y[i]] -=1
      dW += X[i][:,np.newaxis].dot(dscores[np.newaxis,:])
    
    loss /= N
    loss += reg * np.sum(W**2)

    dW /= N
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    # Forward Pass
    scores = X.dot(W)
    # Numerical stability
    scores -= np.max(scores, axis = 1, keepdims = True)
    # Compute probs using softmax function
    softmax = np.exp(scores) / np.sum(np.exp(scores), axis = 1, keepdims = True)
    # cross-entropy loss
    loss = -np.sum(np.log(softmax[np.arange(N), y])) / N
    # Regularization loss
    loss += reg * np.sum(W**2)
    # Backprop
    dscores = softmax
    dscores[np.arange(N), y] -= 1
    dW = X.T.dot(dscores) / N
    # Regularization gradient
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
