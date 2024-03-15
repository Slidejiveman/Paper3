import os
import warnings
import numpy as np 
from lambeq import BobcatParser, AtomicType, IQPAnsatz, remove_cups, TketModel, BinaryCrossEntropyLoss, QuantumTrainer, SPSAOptimizer, Dataset
from discopy.drawing import Equation
from pytket.extensions.qiskit import AerBackend
import matplotlib.pyplot as plt

# see https://cqcl.github.io/lambeq/tutorials/trainer_quantum.html
# This is a simulated quantum sample.
warnings.filterwarnings('ignore')
os.environ['TOKENIZER_PARALLELISM'] = 'true'

# define hyperparameters
BATCH_SIZE = 10
EPOCHS = 100
SEED = 2

# read in datasets from file
def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = int(line[0])
            labels.append([t, 1-t])
            sentences.append(line[1:].strip())
    return labels, sentences

# These files can be found on the lambeq github. Copy their format for original datasets.
# https://github.com/CQCL/lambeq/tree/main/docs/examples/datasets
train_labels, train_data = read_data('./datasets/examples/rp_train_data.txt')
val_labels, val_data = read_data('./datasets/examples/rp_test_data.txt')
# perform test prints to confirm data is present
# print(train_data[:5])
# print(train_labels[:5])

# create and parameterise diagrams
parser = BobcatParser(root_cats=('NP', 'N'), verbose='text')
raw_train_diagrams = parser.sentences2diagrams(train_data, suppress_exceptions=True)
raw_val_diagrams = parser.sentences2diagrams(val_data, suppress_exceptions=True)

# filter and simplify diagrams
train_diagrams = [
    diagram.normal_form()
    for diagram in raw_train_diagrams if diagram is not None
]
val_diagrams =[
    diagram.normal_form()
    for diagram in raw_val_diagrams if diagram is not None
]
train_labels = [
    label for (diagram, label)
    in zip(raw_train_diagrams, train_labels)
    if diagram is not None
]
val_labels = [
    label for (diagram, label)
    in zip(raw_val_diagrams, val_labels)
    if diagram is not None
]
# perform test diagram renders
# train_diagrams[0].draw(figsize=(9, 5), fontsize=12)
# train_diagrams[-1].draw(figsize=(9, 5), fontsize=12)

# Create circuits
# Remove cups to reduce the number of post-selections, which are computationally expensive
ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 0}, n_layers=1, n_single_qubit_params=3)
train_circuits = [ansatz(remove_cups(diagram)) for diagram in train_diagrams]
val_circuits = [ansatz(remove_cups(diagram)) for diagram in val_diagrams]
# perform test circuit draw
# train_circuits[0].draw(figsize=(9, 10))

### For an example of a diagram with and without cups, uncomment the below example
# original_diagram = train_diagrams[0]
# removed_cups_diagram = remove_cups(original_diagram)
# Equation(original_diagram, removed_cups_diagram, symbol='-->').draw(figsize=(9, 6), asymmetry=0.3, fontsize=12)
###

# Instantiate Training Model (not pre-trained in this instance)
all_circuits = train_circuits + val_circuits
backend = AerBackend()
backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 8192
}
model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)
# example of loading a model from a checkpoint
# model = TketModel.from_checkpoint('./logs/best_model.lt', backend_config=backend_config)

# define loss and evaluation metrics
bce = BinaryCrossEntropyLoss()
acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2 # half due to double-counting
eval_metrics = {"acc": acc}
# initialize trainer
trainer = QuantumTrainer(
    model,
    loss_function=bce,
    epochs=EPOCHS,
    optimizer=SPSAOptimizer,
    optim_hyperparams={'a': 0.05, 'c': 0.06, 'A': 0.001*EPOCHS},
    evaluate_functions=eval_metrics,
    evaluate_on_train=True,
    verbose= 'text',
    log_dir='./logs',
    seed=0
)
# create datasets
train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
val_dataset = Dataset(val_circuits, val_labels, shuffle=False)
# train
trainer.fit(train_dataset, val_dataset, early_stopping_interval=10)

# Visualize results and evaluate model on test data using best model from the "logs" folder
fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
ax_tl.set_title('Training set')
ax_tr.set_title('Development set')
ax_bl.set_xlabel('Iterations')
ax_br.set_xlabel('Iterations')
ax_bl.set_ylabel('Accuracy')
ax_tl.set_ylabel('Loss')
colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
range_ = np.arange(1, len(trainer.train_epoch_costs)+1)
ax_tl.plot(range_, trainer.train_epoch_costs, color=next(colours))
ax_bl.plot(range_, trainer.train_eval_results['acc'], color=next(colours))
ax_tr.plot(range_, trainer.val_costs, color=next(colours))
ax_br.plot(range_, trainer.val_eval_results['acc'], color=next(colours))
# mark best model as circle
best_epoch = np.argmin(trainer.val_costs)
ax_tl.plot(best_epoch + 1, trainer.train_epoch_costs[best_epoch], 'o', color='black', fillstyle='none')
ax_tr.plot(best_epoch + 1, trainer.val_costs[best_epoch], 'o', color='black', fillstyle='none')
ax_bl.plot(best_epoch + 1, trainer.train_eval_results['acc'][best_epoch], 'o', color='black', fillstyle='none')
ax_br.plot(best_epoch + 1, trainer.val_eval_results['acc'][best_epoch], 'o', color='black', fillstyle='none')
ax_tr.text(best_epoch + 1.4, trainer.val_costs[best_epoch], 'early stopping', va='center')
plt.show()

# print test accuracy
model.load(trainer.log_dir + '/best_model.lt')
test_acc = acc(model(val_circuits), val_labels)
print('Validation accuracy:', test_acc.item())