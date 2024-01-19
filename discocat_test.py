## DisCoCat uses functors to mp diagrams from the rigid category of pregroup grammars
## to vector space semantics.

## Pregroup grammar I -> T
# noun == n
# adj == nn.l
# transitive verb == n.rsn.l
# adjoints are considered the left and right inverses of a type.
# words are concatenated using the monoidal product and linked using cups.
# a single uncontracted s wire means the sentence is grammatically sound.
# the monoidal unit I is defined as Ty()
from discopy.grammar.pregroup import Cap, Cup, Id, Ty, Word
from lambeq import pregroups

n, s = Ty('n'), Ty('s')
words = [
    Word('she', n),
    Word('goes', n.r @ s @ n.l),
    Word('home', n)
]
cups = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
# * is 'unpacking' a list or a tuple in this sense (https://geekflare.com/python-unpacking-operators/) 
# ** is used to unpack dictionaries
assert Id().tensor(*words) == words[0] @ words[1] @ words[2] 
assert Ty().tensor(*[n.r, s, n.l]) == n.r @ s @ n.l
diagram = Id().tensor(*words) >> cups
# pregroups.draw(diagram)

## Alternative pregroup diagram creation method using lambeq
from lambeq import create_pregroup_diagram
from discopy.grammar.pregroup import Ty
words = [Word('she', n), Word('goes', n.r @ s @ n.l), Word('home', n)] # lambeq method
morphisms = [(Cup, 0, 1), (Cup, 3, 4)]
diagram = create_pregroup_diagram(words, Ty('s'), morphisms)
# diagram.draw()
same_words = Word('she', n) @ Word('goes', n.r @ s @ n.l) @ Word('home', n) # discopy method
same_diagram = same_words.cup(0, 1).cup(1, 2) # another alternative using discopy
# same_diagram.draw()
# Implicit swap introduction by using .cup() on non-adjacent qubits
n, s, p = map(Ty, "nsp")
words = Word('A', n @ p) @ Word('V', n.r @ s @ n.l) @ Word('B', p.r @ n)
# words.cup(1, 5).cup(0, 1).cup(1, 2).draw()

## Note: only diagrams of form word @ ... @  word >> cups_and_swaps can be drawn using
##       lambeq.pregroups.draw(). Functors and normal forms typically mess this up.
##       In these cases, use monoidal.Diagram.draw() instead to avoid ValueError.
from discopy.drawing import Equation
from discopy import monoidal
from pytest import raises
# In the original diagram, words appear before the cups
# print('Before normal form:', ', '.join(map(str, diagram.boxes)))
diagram_nf = monoidal.Diagram.normal_form(diagram)
# print('After normal form:', ', '.join(map(str, diagram_nf.boxes)))
# Equation(diagram, diagram_nf, symbol='->').draw(figsize=(10, 4))
# In the normalised diagram, boxes are not in the right order anymore,
# so cannot be drawn using pregroups.draw()
with raises(ValueError):
    pregroups.draw(diagram_nf)

## Functors: structure preserving transformation between different monoidal categories
# monoidal structure of objects is preserved
# adjoints are preserved
# monoidal structure of morphisms is preserved
# compositional structure of morphisms is preserved
# In free monoidal categoryl, applying a functor to a diagram amounts to simply providing a mappint
# for each generating object and morphism.
# In DisCoPy, a functor is defined by passing mappings (dictionaries or functions) as arguments
# ob and ar to the Functor class.
# lambeq's pipeline is implemented entirely based on the concept of Functors
# lambeq.CCGParser -> pregroup diagram -> lambeq.Rewriter -> simpler pregroup diagram -> (next choose a or b)
# (a:lambeq.TensorAnsatz -> tensor diagram/network) OR (b:lambeq.CircuitAnsatz -> quantum circuit)

## Example 1: "Very" Functor
# This functor adds the word "very" before every adjective in a DiscoCat diagram.
# use pregroup.Functor since we are mapping from a pregroup.Diagram to another pregroup.Diagram
# "very" is typed as (n @ n.l) @ (n @ n.l).l = n @ n.l @ n.l.l @ n.l
from lambeq import BobcatParser
from discopy.grammar.pregroup import Diagram, Functor
parser = BobcatParser(verbose='suppress') 
# determiners have the same type as adjectives
# but shouldn't add 'very' behind them
determiners = ['a', 'the', 'my', 'his', 'her', 'their']
# type for an adjective (this is a good example for  me to reference for an h type)
adj = n @ n.l
very = Word('very', adj @ adj.l)
cups = Diagram.cups(adj.l, adj)

# Methods
def very_ob(ty):
    return ty

def very_ar(box):
    if box != very:
        if box.name not in determiners:
            if box.cod == adj:
                return very @ box >> Id(adj) @ cups
    return box

# procedure
very_functor = Functor(ob=very_ob, ar=very_ar)
diagram = parser.sentence2diagram('a big bad wolf')
new_diagram = very_functor(diagram)
# Equation(diagram, new_diagram, symbol='->').draw(figsize=(10, 4))

## Example 2: Twist functor [5]
from discopy.grammar.pregroup import Category, Diagram, factory, Functor, Swap

## Classes
@factory
class TwistedDiagram(Diagram):
    @classmethod
    def cups(cls, left, right):
        swaps = Diagram.swap.__func__(cls, left, right)
        cups = Diagram.cups.__func__(cls, right, left)
        return swaps >> cups
    
    @classmethod
    def caps(cls, left, right):
        return cls.cups(left, right).dagger()
# add the TwistedDiagram interface to the others
class TwistedWord(Word, TwistedDiagram): ...
class TwistedSwap(Swap, TwistedDiagram): ...
class TwistedCup(Cup, TwistedDiagram): ...
class TwistedCap(Cap, TwistedDiagram): ...

TwistedDiagram.braid_factory = TwistedSwap
TwistedDiagram.cup_factory = TwistedCup
TwistedDiagram.cap_factory = TwistedCap

twist_functor = Functor(
    ob=lambda ty: ty,
    ar=lambda word: TwistedWord(word.name, word.cod),
    cod=Category(Ty, TwistedDiagram)
)
diagram = parser.sentence2diagram('This is twisted')
new_diagram = twist_functor(diagram)
# pregroups.draw(diagram)
# pregroups.draw(new_diagram)
snake = Id(n) @ Cap(n.r, n) >> Cup(n, n.r) @ Id(n)
# NOTE:  twisting "is" and "twisted" together is not a fucntorial operation
# so it cannot be implemented using a rigid.Functor.
# Equation(twist_functor(snake), Id(n)).draw(figsize=(4, 2))

## Classical DisCoCat: Tensor networks (FVect)
# Compact Closed Category because A.l = A.r = A*
# Objects are defined with the Dim class and morphisms with the Box class.
# The concrete value of the tensor is pa ssed to the data attribute as an unshaped list.
# DisCoPy will reshape it later based on input and output dimensions.
# tensor.Tensor computes tensor contractions directly while tensor.Diagram waits on the eval() method
from discopy.tensor import Box, Dim, Id, Tensor
# NOTE: Dim(1) is the unit object, so disappears when tensored with another Dim
# print(f'{Dim(1) @ Dim(2) @ Dim(3)=}')
id_box = Box('Id Box', Dim(2), Dim(2), data=[1,0,0,1])
id_tensor = Tensor([1,0,0,1], Dim(2), Dim(2))
# The actual values of the id_box and id_tensor are equal
assert (id_box.array == id_tensor.array).all()
# print(f'{id_box.eval()=}')
import numpy as np
f_box = Box('f Box', Dim(2, 2), Dim(2), data=range(8))
f_tensor = Tensor(range(8), Dim(2, 2), Dim(2))
combined_diagram = id_box @ Id(Dim(2)) >> f_box
combined_tensor = id_tensor @ Tensor.id(Dim(2)) >> f_tensor
# tensor diagram evaluates to the tensor
assert combined_diagram.eval(dtype=np.int32) == combined_tensor
# combined_diagram.draw(figsize=(4, 2))
# print(combined_tensor)
# In FVect, cups, caps, and swaps take on concrete values as tensors as well.
# NOTE: This is extremely helpful when you want to understand what is going on mathematically.
# print(Tensor.cups(Dim(3), Dim(3)).array)
# print(Tensor.swap(Dim(2), Dim(2)).array)
# To implement a functor from pregroup.Diagram to tensor.Tensor, use a tensor.Functor with
# dom=pregroup.Category(). This will automatically contract the resulting tensor network.
from discopy.tensor import Functor
import numpy as np
## Methods
def one_ob(ty):
    dims = [2] * len(ty)
    return Dim(*dims) # does Dim(2,2,...) via unpacking

def one_ar(box):
    dom = one_ob(box.dom)
    cod = one_ob(box.cod)
    tensor = np.ones((dom @ cod).inside)
    print(f'"{box}" becomes')
    print(tensor)
    return tensor

## precedural code
one_functor = Functor(ob=one_ob, ar=one_ar, dom=Category())
# print(one_functor(diagram))

## Quantum DisCoCat: Quantum Circuits
# sends diagrams in the category of pregroup derivations to circuits in FHilb, which is
# a compact closed monoidal category with Hilbert Spaces (e.g. C^(2^n)) as objects and unitary maps between
# Hilbert spaces as morphisms.
# Objects are generated with the quantum.circuit.Ob class and morphisms with quantum.gates.
# rotation values are 0 to 1 instead of 0 to 2π.
# evaluate with tensor contraction via the "eval()" method or export to pytket using "to_tk()"
from discopy.quantum import qubit, Id
from discopy.quantum.gates import CX, Rz, X
circuit = Id(4)
circuit >>= Id(1) @ CX @ X
circuit >>= CX @ CX
circuit >>= Rz(0.1) @ Rz(0.2) @ Rz(0.3) @ Rz(0.4)
# from discopy 0.4.1 can do:
same_circuit = (Id(4).CX(1, 2).X(3).CX(0, 1).CX(2, 3).Rz(0.1, 0).Rz(0.2, 1).Rz(0.3, 2).Rz(0.4, 3))
assert circuit == same_circuit
# circuit.draw()
# print(circuit.to_tk())
# Swaps are used to permute the wires when you want to apply multi-qubit gates to non-adjacent wires.
from discopy.quantum import Circuit
from discopy.quantum.gates import SWAP
# to apply a CNOT on qubits 2 and 0: NOTE: Notice the symmetric structure
circuit1 = Id(3)
circuit1 >>= SWAP @ Id(1)
circuit1 >>= Id(1) @ SWAP
circuit1 >>= Id(1) @ CX
circuit1 >>= Id(1) @ SWAP
circuit1 >>= SWAP @ Id(1)
# or you can do
perm = Circuit.permutation([2, 0, 1])
circuit2 = perm[::-1] >> Id(1) @ CX >> perm
assert circuit1 == circuit2
# circuit1.draw(figsize=(3, 3))
# no swpas are introduced when converting to tket
# print(circuit1.to_tk())
# Since discopy 0.4.0, long-ranged controlled gates have been added. YAY!
from discopy.quantum import Controlled, Rz, X
# (Controlled(Rz(0.5), distance=2) >> Controlled(X, distance=-2)).draw(figsize=(3, 2))
# Controlled(Controlled(X), distance=2).draw(figsize=(3, 2))
# All of the above circuits are "pure" because they consist strictly of unitaries.
# Pure circuits can be evaluated locally to return a Tensor.
# Circuits that contain Discard and Measure are considered "mixed" and return Channel
# when they are evaluated because they are not unitaries but rather classical-quantum maps.
from discopy.quantum import Discard, Measure, Ket, Bra
from discopy.quantum.channel import C, Q
# print((C(Dim(2)) @ Q(Dim(2, 3)) @ C(Dim(2))))
# print(Discard().eval())
# print(Measure().eval())
# print(Ket(0).eval())
# circuits that have measurements in them are no longer unitary and return CQMaps
# print((Ket(0) >> Measure()).eval())
# pure circuits can be coerced to evaluate into a CQMap by using ".eval(mixed=True)"
# NOTE: the tensor order of a CQMap is doubled compared to that of a simple Tensor.
# print(CX.eval().array.shape)
# print(CX.eval(mixed=True).array.shape)  
# To implement a functor from rigid.Diagram to quantum.Circuit, use a quantum.circuit.Functor
from discopy.quantum.circuit import Functor, Id
## Methods
def cnot_ob(ty):
    # this implicitely maps all rigid types to 1 qubit
    return qubit ** len(ty)

def cnot_ar(box):
    dom = len(box.dom)
    cod = len(box.cod)
    width = max(dom, cod)
    circuit = Id(width)
    for i in range(width - 1):
        circuit >>= Id(i) @ CX @ Id(width - i - 2)
    # Add bras (post-selection) and Kets (states)
    # to get a circuit with the right amount of
    # input and output wires
    if cod <= dom:
        circuit >>= Id(cod) @ Bra(*[0]*(dom - cod))
    else:
        circuit <<= Id(dom) @ Ket(*[0]*(cod - dom))
    return circuit

## Prodecure
cnot_functor = Functor(ob=cnot_ob, ar=cnot_ar, dom=Category())
diagram.draw()
cnot_functor(diagram).draw(figsize=(8, 8))
