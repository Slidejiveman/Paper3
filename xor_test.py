from torch import nn
from lambeq import AtomicType, BobcatParser, IQPAnsatz, PennyLaneModel, remove_cups, Dataset
from itertools import combinations
import torch
import random
import numpy as np

# hyperparameters
BATCH_SIZE = 50
EPOCHS = 100
LEARNING_RATE = 0.1
SEED = 2
TRAIN_SAMPLES, DEV_SAMPLES, TEST_SAMPLES = 300, 200, 200
FOOD_IDX, IT_IDX = 0, 6

# setup seeds
torch.manual_seed(SEED)
random.seed(SEED)

## Classes 
# Using the PennyLaneModel as a base, add the XOR function and forwarding.
class XORSentenceModel(PennyLaneModel):
    def __init__(self, **kwargs):
        PennyLaneModel.__init__(self, **kwargs)
        self.xor_net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())
        
    def forward(self, diagram_pairs):
        first_d, second_d = zip(*diagram_pairs)
        evaluated_pairs = torch.cat((self.get_diagram_output(first_d),
                                     self.get_diagram_output(second_d)), dim=1)
        evaluated_pairs = 2 * (evaluated_pairs - 0.5)
        return self.xor_net(evaluated_pairs)

## methods
# read data in from files.
def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = float(line[0])
            labels.append([t, 1-t])
            sentences.append(line[1:].strip())
    return labels, sentences
train_labels, train_data = read_data('./datasets/examples/mc_train_data.txt')
dev_labels, dev_data = read_data('./datasets/examples/mc_dev_data.txt')
test_labels, test_data = read_data('./datasets/examples/mc_test_data.txt')

# this is used to construct a dataset of pairs of diagrams
def make_pair_data(diagrams, labels):
    pair_diags = list(combinations(diagrams, 2))
    pair_labels = [int(x[0] == y[0]) for x, y in combinations(labels, 2)]
    return pair_diags, pair_labels

# accuracy function
def accuracy(circs, labels):
    predicted = model(circs)
    return (torch.round(torch.flatten(predicted)) == torch.Tensor(labels)).sum().item()/len(circs)

##
# create, parameterize, and simplify diagrams
reader = BobcatParser(verbose='text')
raw_train_diagrams = reader.sentences2diagrams(train_data)
raw_dev_diagrams = reader.sentences2diagrams(dev_data)
raw_test_diagrams = reader.sentences2diagrams(test_data)
train_diagrams = [remove_cups(diagram) for diagram in raw_train_diagrams]
dev_diagrams = [remove_cups(diagram) for diagram in raw_dev_diagrams]
test_diagrams = [remove_cups(diagram) for diagram in raw_test_diagrams]

# create circuits
ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3)
train_circuits = [ansatz(diagram) for diagram in train_diagrams]
dev_circuits = [ansatz(diagram) for diagram in dev_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]

# make pairs using circuits
train_pair_circuits, train_pair_labels = make_pair_data(train_circuits, train_labels)
dev_pair_circuits, dev_pair_labels = make_pair_data(dev_circuits, dev_labels)
test_pair_circuits, test_pair_labels = make_pair_data(test_circuits, test_labels)
train_pair_circuits, train_pair_labels = (zip(*random.sample(list(zip(train_pair_circuits, train_pair_labels)), TRAIN_SAMPLES)))
dev_pair_circuits, dev_pair_labels = (zip(*random.sample(list(zip(dev_pair_circuits, dev_pair_labels)), DEV_SAMPLES)))
test_pair_circuits, test_pair_labels = (zip(*random.sample(list(zip(test_pair_circuits, test_pair_labels)), TEST_SAMPLES)))

# initialise model
all_pair_circuits = (train_pair_circuits + dev_pair_circuits + test_pair_circuits)
a, b = zip(*all_pair_circuits)
model = XORSentenceModel.from_diagrams(a + b) 
model.initialise_weights()
model = model
train_pair_dataset = Dataset(train_pair_circuits, train_pair_labels, batch_size=BATCH_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train and Log Accuracies
best = {'acc': 0, 'epoch': 0}
for i in range(EPOCHS):
    epoch_loss = 0
    for circuits, labels in train_pair_dataset:
        optimizer.zero_grad()
        predicted = model(circuits)
        loss = torch.nn.functional.binary_cross_entropy(torch.flatten(predicted), torch.Tensor(labels))
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    if i % 5 == 0:
        dev_acc = accuracy(dev_pair_circuits, dev_pair_labels)
        print()
        print('Epoch: {}'.format(i))
        print('Train loss: {}'.format(epoch_loss))
        print('Dev acc: {}'.format(dev_acc))
    if dev_acc > best['acc']:
        best['acc'] = dev_acc
        best['epoch'] = i
        model.save('xor_model.lt')
    elif i - best['epoch'] >= 10:
        print()
        print('Early stopping')
        break
if best['acc'] > accuracy(dev_pair_circuits, dev_pair_labels):
    model.load('xor_model.lt')
    model = model
print('Final test accuracy: {}'.format(accuracy(test_pair_circuits, test_pair_labels)))

# Analyze the internal representation of the model
xor_labels = [[1,0,1,0], [0,1,0,1], [1,0,0,1], [0,1,1,0]]
xor_tensors = torch.tensor(xor_labels).float()
print(model.xor_net(xor_tensors).detach().numpy())

## Look at output
# Food
symbol_weight_map = dict(zip(model.symbols, model.weights))
print(test_data[FOOD_IDX])
p_circ = test_circuits[FOOD_IDX].to_pennylane(probabilities=True)
p_circ.initialise_concrete_params(symbol_weight_map)
unnorm = p_circ.eval().detach().numpy()
print(unnorm / np.sum(unnorm))

# IT
print(test_data[IT_IDX])
p_circ = test_circuits[IT_IDX].to_pennylane(probabilities=True)
p_circ.initialise_concrete_params(symbol_weight_map)
unnorm = p_circ.eval().detach().numpy()
print(unnorm / np.sum(unnorm))