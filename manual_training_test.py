import warnings
warnings.filterwarnings('ignore') # Ignore warnings

from jax import numpy as np, jit, grad
from lambeq import BobcatParser, AtomicType, SpiderAnsatz
from sympy import default_sort_key
import numpy

from discopy.tensor import set_backend, Dim
set_backend('jax') # Tell discopy to use JAX's version of numpy

numpy.random.seed(0) # Fix the seed
np.random = numpy.random

N = AtomicType.NOUN
S = AtomicType.SENTENCE

## Methods (NOTE: These examples use tensors and not quantum circuits.)
# read file
def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data, targets = [], []
    for ln in lines:
        t = int(ln[0])
        data.append(ln[1:].strip())
        targets.append(np.array([t, not(t)], dtype=np.float32))
    return data, np.array(targets)
# define sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# define loss funciton 
def loss(tensors):
    # lambdify to replace symbols with concrete numpy arrays
    np_circuits = [c.lambdify(*vocab)(*tensors) for c in train_circuits]
    # Compute predictions
    predictions = sigmoid(np.array([c.eval(dtype=float).array for c in np_circuits]))
    # Binary cross-entropy loss
    cost = -np.sum(train_targets * np.log2(predictions)) / len(train_targets)
    return cost

## Input data
train_data, train_targets = read_data('./datasets/examples/mc_train_data.txt')
test_data, test_targets = read_data('./datasets/examples/mc_test_data.txt')
# print(train_data[:10]) # Sentences
# print(train_targets[:10]) # 2D arrays

## Create and parameterize diagrams
parser = BobcatParser(verbose='suppress')
train_diagrams = parser.sentences2diagrams(train_data)
test_diagrams = parser.sentences2diagrams(test_data)
# train_diagrams[0].draw(figsize=(8,4), fontzize=13)
# Create an ansatz by assinging 2 dimensions to both noun and sentence spaces
ansatz = SpiderAnsatz({N: Dim(2), S: Dim(2)})
train_circuits = [ansatz(d) for d in train_diagrams]
test_circuits = [ansatz(d) for d in test_diagrams]
all_circuits = train_circuits + test_circuits
# all_circuits[0].draw(figsize=(8, 4), fontsize=13)

## Creating a vocabulary
vocab = sorted({sym for circ in all_circuits for sym in circ.free_symbols}, key=default_sort_key)
tensors = [np.random.rand(w.size) for w in vocab]
# print(tensors[0])

## Training: define loss function
training_loss = jit(loss)
gradient = jit(grad(loss))

## Train
training_losses = []
epochs = 90
for i in range(epochs):
    gr = gradient(tensors)
    for k in range(len(tensors)):
        tensors[k] = tensors[k] - gr[k] * 1.0
    training_losses.append(float(training_loss(tensors)))
    if (i + 1) % 10 == 0:
        print(f"Epoch {i + 1} - loss {training_losses[-1]}")
        
## Evaluate
# Testing
np_test_circuits = [c.lambdify(*vocab)(*tensors) for c in test_circuits]
test_predictions = sigmoid(np.array([c.eval(dtype=float).array for c in np_test_circuits]))
hits = 0
for i in range(len(np_test_circuits)):
    target = test_targets[i]
    pred = test_predictions[i]
    if np.argmax(target) == np.argmax(pred):
        hits += 1
print("Accuracy on test set:", hits / len(np_test_circuits))

## Working with quantum circuits
# is similar, but the following must be changed:
# 1. The parameterisable part of the circuit is an array of parameters as described in Section Circuit symbols
#    instead of tensors associated with words.
# 2. The optimization taks place on quantum hardware. Standard automatic differentiation cannot be used.
#    Instead, an alternative technique such as Simultaneous Perturbation Stochastic Approximation can be used.