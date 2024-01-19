## Readers are the base class that provides the ability to convert a sentence to a rigid category.
## Reader example: "Comb" Reader
## The comb reader is not approrpiate for classical experiments, but it will work without a problem
## on a quantum computer since we don't have to worry about huge dimensionality in tensors.
from lambeq import AtomicType, Reader
from discopy.grammar.pregroup import Box, Diagram, Id, Word
from lambeq.core.utils import SentenceType

N = AtomicType.NOUN

## Class
class CombReader(Reader):
    def sentence2diagram(self, sentence):
        #"tensor" in the following line refers to the monoidal product, not a physical tensor
        words = Id().tensor(*[Word(w, N) for w in sentence.split()]) # tokenizes the words
        layer = Box('LAYER', words.cod, N) # in an actual implementation, all sentences should share the same layer.
        return words >> layer
    
## Procedure
diagram = CombReader().sentence2diagram('John gave Mary a flower')
# diagram.draw()

## Creating rewrite rules
from lambeq import BobcatParser
parser = BobcatParser(verbose='text')
d = parser.sentence2diagram('The food is fresh')
## Negation Functor
## Use the SimpleRewriteRule class to facilitate the creation of simple rewrite rules
## without the need to define a new RewriteRule class from scratch. It finds words with
## "cod" and "name" in list "words" and replaces their boxes with the diagram in "template".
from lambeq import AtomicType, SimpleRewriteRule, Rewriter
from discopy.drawing import Equation

N = AtomicType.NOUN
S = AtomicType.SENTENCE
adj = N @ N.l
NOT = Box('NOT', S, S)
negation_rewrite = SimpleRewriteRule(cod=N.r @ S @ S.l @ N, 
                                     template=SimpleRewriteRule.placeholder(N.r @ S @ S.l @ N) >> Id(N.r) @ NOT @ Id(S.l @ N),
                                     words=['is', 'was', 'has', 'have'])
not_d = Rewriter([negation_rewrite])(d)
# Equation(d, not_d, symbol='->').draw(figsize=(14, 4))

## Reader Example: Past Functor
from lambeq import RewriteRule

## Class
class PastRewriteRule(RewriteRule):
    mapping ={ 'is': 'was', 'are': 'were', 'has': 'had' }
    def matches(self, box):
        return box.name in self.mapping
    def rewrite(self, box):
        new_name = self.mapping[box.name]
        return type(box)(name=new_name, dom=box.dom, cod=box.cod)
    
## Procedure (NOTE: Past tense does not change the grammatical links in the quantum network.)
past_d = Rewriter([PastRewriteRule()])(d)
# Equation(d, past_d, symbol='->').draw(figsize=(14, 4))

## Creating Ansatze
# These are implemented by extending the CircuitAnsztz class for the quantum pipeline.
# TensorAnsatz is used for the classical pipeline. Both extend the BaseAnsatz.
# Once instantiated, it can be used as a functor to convert diagrams to circuits or tensors.
# Initialized with an "ob_map" argument, which is a dictionary that maps a rigid type to the
# number of qubits in the quantum case, or to the dimension size for the classical case.
# All you need to provide is provide the mapping from rigid boxes to diagrams.
# 1. Obtain the label of the box using the _summarise_box method.
# 2. Apply the functor to the domain and the codomain of the box.
# 3. Construct and return an ansatz with new domain and codomain
## CircuitAnsatz Example: "Real-valued" ansatz
# This ansatz always returns a tensor with real-valued entries, since the ansatz is constructed using
# only the CNOT and Y rotation gates, which both implement real-valued unitaries.
# The CircuitAnsatz provides functionality to add postselections or discards to ensure the domains and
# codomains for the boxes match. All that is required is to provide a function to generate the circuit
# within the box.
from discopy.quantum.circuit import Functor, Id
from discopy.quantum.gates import Bra, CX, Ket, Ry
from lambeq import CircuitAnsatz

## Methods
def real_ansatz_circuit(n_qubits, params):
    circuit = Id(n_qubits)
    n_layers = params.shape[0] - 1
    for i in range(n_layers):
        syms = params[i]
        # adds a layer of Y rotations
        circuit >>= Id().tensor(*[Ry(sym) for sym in syms])
        # adds a ladder of CNOTs
        for j in range(n_qubits - 1):
            circuit >>= Id(j) @ CX @ Id(n_qubits - j - 2)
    # adds a final layer of Y rotations
    circuit >>= Id().tensor(*[Ry(sym) for sym in params[-1]])
    return circuit

## Class
class RealAnsatz(CircuitAnsatz):
    def __init__(self, ob_map, n_layers, n_single_qubit_params = 1, discard = False):
        super().__init__(ob_map, n_layers, n_single_qubit_params, real_ansatz_circuit, discard, [Ry, ])
    def params_shape(self, n_qubits):
        return (self.n_layers + 1, n_qubits)
    
## Procedure
real_d = RealAnsatz({N: 1, S: 1}, n_layers=2)(d)
# real_d.draw(figsize=(12, 10))

## TensorAnsatz example: "Positive" ansatz
from lambeq import TensorAnsatz, Symbol
from discopy import rigid, tensor
from discopy.tensor import Dim
from functools import reduce
import math

## Class
class PositiveAnsatz(TensorAnsatz):
    def _ar(self, box):
        # step 1: obtain label
        label = self._summarise_box(box)
        # step 2: map domain and codomain
        dom, cod = self.functor(box.dom), self.functor(box.cod)
        # step 3: construct and return ansatz
        name = self._summarise_box(box)
        syms = Symbol(name, math.prod(dom.inside), math.prod(cod.inside))
        return tensor.Box(box.name, dom, cod, syms ** 2) # squares the tensors so they are always positive
    
## Pocedure
ansatz = PositiveAnsatz({N: Dim(2), S: Dim(2)})
positive_d = ansatz(d)
positive_d.draw()

## Concretize symbols as tensors with lamdify and show form.
import numpy as np
from sympy import default_sort_key

syms = sorted(positive_d.free_symbols, key=default_sort_key)
sym_dict = {k: -np.ones(k.size) for k in syms}
subbed_diagram = positive_d.lambdify(*syms)(*sym_dict.values())
print(subbed_diagram.eval())