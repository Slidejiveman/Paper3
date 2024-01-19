import warnings
warnings.filterwarnings('ignore')

from lambeq import AtomicType, BobcatParser, TensorAnsatz, IQPAnsatz
from discopy.tensor import Dim 
from sympy import default_sort_key
import numpy as np

## Introduction
# Define atomic types
N = AtomicType.NOUN
S = AtomicType.SENTENCE

# Parse a sentence
parser = BobcatParser(verbose='suppress')
diagram = parser.sentence2diagram('John walks in the park')

# Apply a tensor ansatz & draw the diagram
ansatz = TensorAnsatz({N: Dim(4), S:Dim(2)})
tensor_diagram = ansatz(diagram)
# tensor_diagram.draw(figsize=(12, 5), fontsize=12)

# Access symbols of the diagram to see a list of symbols and also their size in dimensions
print('Diagram symbols')
print(tensor_diagram.free_symbols)
print([(s, s.size) for s in tensor_diagram.free_symbols])

## Circuit Symbols. These are associated with rotation angles on qubits and not a tensor size. (Use "1")
iqp_ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
circuit = iqp_ansatz(diagram)
# circuit.draw(figsize=(12, 8), fontsize=12) #it can take a few seconds for the circuit to display.
print('Circuit symbols')
print(circuit.free_symbols) # no need to print out size for circuits

## From symbols to tensors
# compare the symbol with the tensor representation
parameters = sorted(tensor_diagram.free_symbols, key=default_sort_key)
tensors = [np.random.rand(p.size) for p in parameters]
print(tensors[0])
tensor_diagram_np = tensor_diagram.lambdify(*parameters)(*tensors)
print("Before lambdify:", tensor_diagram.boxes[0].data)
print("After lambdify:", tensor_diagram_np.boxes[0].data)
# contract the tensor network
result = tensor_diagram_np.eval(dtype=float)
print(result)
print(result.array) # 2D array because dimension of the S type was set to 2 via TensorAnsatz