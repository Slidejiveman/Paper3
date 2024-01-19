import torch
import random
import numpy as np
from lambeq import BobcatParser, remove_cups, AtomicType, IQPAnsatz, PennyLaneModel, Dataset, PytorchTrainer
import pennylane as qml
import matplotlib.pyplot as plt

# hyperparameters
BATCH_SIZE = 10
EPOCHS = 15
LEARNING_RATE = 0.1
SEED = 42
# setup seeds
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# define a method to read the data in from files.
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
# test file read by printing the first 5 from a data array
#print(train_data[:5])
#print(train_labels[:5])

# create, parameterize, and simplify diagrams
reader = BobcatParser(verbose='text')
raw_train_diagrams = reader.sentences2diagrams(train_data)
raw_dev_diagrams = reader.sentences2diagrams(dev_data)
raw_test_diagrams = reader.sentences2diagrams(test_data)
train_diagrams = [remove_cups(diagram) for diagram in raw_train_diagrams]
dev_diagrams = [remove_cups(diagram) for diagram in raw_dev_diagrams]
test_diagrams = [remove_cups(diagram) for diagram in raw_test_diagrams]
# test diagram display
#train_diagrams[0].draw()

# create circuits
ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1, n_single_qubit_params=3)
train_circuits = [ansatz(diagram) for diagram in train_diagrams]
dev_circuits = [ansatz(diagram) for diagram in dev_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]
# test display of a circuit
#train_circuits[0].draw(figsize=(6, 8))

# Training
all_circuits = train_circuits + dev_circuits + test_circuits
# this is the default backend_config, but it is listed here for educational purposes
# probabilities=True cau ses the model to output probabilities instead of quantum states 
# just like real quantum computers.
backend_config = {'backend': 'default.qubit'} # default PennyLane quantum simulator
model = PennyLaneModel.from_diagrams(all_circuits, probabilities=True, normalize=True, backend_config=backend_config)
model.initialise_weights()

# ## This section can be updated with different API tokens and different devices.
# ## Use this example for real experiments. It is too slow for a tutorial.
# qml.default_config['qiskit.ibmq.ibmqx_token'] = 'd5c8a439805bafd35390cede4d49fe4a3314f4a330ca9aa86b6fc8c7b72797ee54272fee02c31a1451aa61c753eab7cacd4bd70c220e76a5cd2927693eb4fa36'
# qml.default_config.save(qml.default_config.path)
# backend_config = {'backend': 'qiskit.ibmq', 'device': 'ibmq_manila', 'shots': 1000} 
# # initialize quantum model (q_model)
# q_model = PennyLaneModel.from_diagrams(all_circuits, probabilities=True, normalize=True, backend_config=backend_config)
# q_model.initialise_weights()

# Create datasets
train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
val_dataset = Dataset(dev_circuits, dev_labels)

## Define metrics required for PytorchTrainer
# Define evaluation metric. This is accuracy.
def acc(y_hat, y):
    return (torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)).sum().item() / len(y)
# Define loss metric. This is the Mean-Squared Error
def loss(y_hat, y):
    return torch.nn.functional.mse_loss(y_hat, y)

# Train!
trainer = PytorchTrainer(model=model, loss_function=loss, optimizer=torch.optim.Adam, learning_rate=LEARNING_RATE, epochs=EPOCHS,
                         evaluate_functions={'acc': acc}, evaluate_on_train=True, use_tensorboard=False, verbose='text', seed=SEED)
trainer.fit(train_dataset, val_dataset)

# Visualize the results
fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
ax_tl.set_title('Training set')
ax_tr.set_title('Development set')
ax_bl.set_xlabel('Iterations')
ax_br.set_xlabel('Iterations')
ax_bl.set_ylabel('Accuracy')
ax_tl.set_ylabel('Loss')
colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
range_ = np.arange(1, trainer.epochs+1)
ax_tl.plot(range_, trainer.train_epoch_costs, color=next(colours))
ax_bl.plot(range_, trainer.train_eval_results['acc'], color=next(colours))
ax_tr.plot(range_, trainer.val_costs, color=next(colours))
ax_br.plot(range_, trainer.val_eval_results['acc'], color=next(colours))
pred = model(test_circuits)
labels = torch.tensor(test_labels)
print('Final test accuracy: {}'.format(acc(pred, labels)))
#plt.show()

### Standard PyTorch model example
smodel = PennyLaneModel.from_diagrams(all_circuits)
smodel.initialise_weights()
optimizer = torch.optim.Adam(smodel.parameters(), lr=LEARNING_RATE)
best = {'acc': 0, 'epoch': 0}

# define the accuracy metric
def accuracy(circs, labels):
    probs = smodel(circs)
    return (torch.argmax(probs, dim=1) == torch.argmax(torch.tensor(labels), dim=1)).sum().item() / len(circs)

# training loop
for i in range(EPOCHS):
    epoch_loss = 0
    for circuits, labels in train_dataset:
        optimizer.zero_grad()
        probs = smodel(circuits)
        loss = torch.nn.functional.mse_loss(probs, torch.tensor(labels))
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    if i % 5 == 0:
        dev_acc = accuracy(dev_circuits, dev_labels)
        print()
        print('Epoch: {}'.format(i))
        print('Train loss: {}'.format(epoch_loss))
        print('Dev acc: {}'.format(dev_acc))
        if dev_acc > best['acc']:
            best['acc'] = dev_acc
            best['epoch'] = i
            smodel.save('smodel.lt')
        elif i - best['epoch'] >= 10:
            print('Early stopping')
            break
if best['acc'] > accuracy(dev_circuits, dev_labels):
    smodel.load('smodel.lt')
print('Standard model: Final test accuracy: {}'.format(accuracy(test_circuits, test_labels)))