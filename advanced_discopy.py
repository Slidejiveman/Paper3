from discopy.cat import Arrow, Box, Id, Ob

## Free Categories
# Objects
A, B, C, D = map(Ob, 'ABCD')

# Morphisms
f = Box('f', A, B)
g = Box('g', B, C)
h = Box('h', C, D)

# the codomain of f and the domain of g match, so f and g compose
g << f #(read as "g after f")
assert f.cod == g.dom == B
assert g << f == f >> g # ("g after f" is equivalent to "f followed by g")

# associativity
assert h << (g << f) == h << g << f == (h << g) << f

# identity
assert f << Id(A) == f ==  Id(B) << f

# only arrows that 'type-check' can be composed
# arrows behave like List[Box] and can be indexed, sliced, and reversed.
# Reversing a morphism is equivalent to performing the dagger operation. (QM and Linear Algebra)
arrow = h << g << f
assert arrow == f >> g >> h
assert arrow == Arrow(inside=(f, g, h), dom=A, cod=D)
print(arrow)
print(f'Indexing:', arrow[0])
print(f'Slicing:', arrow[1:])
print(f'Reversing (dagger):', arrow[::-1])

## Monoidal Categories
# Adds the monoidal product (x inside a circle) and the monoidal unit "I"
# Objects can be combined to return another object: A @ B
# Morphisms can be combined to return another morphism
# the monoidal product is associative on objects
# the monoidal product is associative on morphisms
# the monoidal unit is the identiy on objects
# the monoidal unit is the identity on morphisms
from discopy.monoidal import Box, Id, Ty

A, B, C = Ty('A'), Ty('B'), Ty('C')
f = Box('f', A, B)
g = Box('g', B, C)
h = Box('h', B, A)

# combining objects
A @ B

# combining arrows
f @ g

# associativity
assert (A @ B) @ C == A @ B @ C == A @ (B @ C)
assert (f @ g) @ h == f @ g @ h == f @ (g @ h)

# monoidal unit
assert A @ Ty() == A == Ty() @ A # objects
assert f @ Id(Ty()) == f == Id(Ty()) @ f # morphisms

# Graphical Calculus for Monoidal Categories
x = Box('x', A, A)
y = Box('y', A @ A, B)
diagram = x @ Id(A) >> y
# print(repr(diagram))
# diagram.draw(figsize=(5, 3))

# Ty operations
t = A @ B @ C
print()
print(t)
print(repr(t))
print(f'Indexing:', t[0])
print(f'Slicing:', t[1:])
print(f'Reversing:', t[::-1])
# It is often desira ble to select a single object from a compount Ty and use it as a type
# since a monoidal.Box expects its domain and codomain to be types (Ty) rather than Objects (Ob)
# Do this with a 1-element slice.
print('Indexing (Ob):', repr(t[0]))
print('Indexing (Ty):', repr(t[0:1])) # This is a trick to get a Ty with a single Ob.

# Monoidal.Diagram operations
print()
print(diagram)
print(f'Indexing:', diagram[0])
print(f'Slicing:', diagram[1:])
print(f'Reversing (dagger):', diagram[::-1])
# Diagram equations
from discopy.drawing import Equation
# print('\nDagger operation:')
# boxes are drawn as trapeziums to demonstrate the reflection along the horizontal axis
# Equation(diagram, diagram[::-1], symbol='->').draw(figsize=(8, 3), asymmetry=0.2)
# Diagram internal representation
from discopy.monoidal import Diagram
offset_diagram = Diagram.decode(dom=Ty('A', 'A'), cod=Ty('B'), 
                                boxes=[Box('x', Ty('A'), Ty('A')), Box('y', Ty('A', 'A'), Ty('B'))],
                                offsets=[1, 0])
# offset_diagram.draw(figsize=(5, 3))

## Symmetric Monoidal Category (has swap)
from discopy.symmetric import Diagram, Swap
# Swap(A, B).draw(figsize=(1, 1))
# Diagram.swap(A @ B, C).draw(figsize=(2, 2))

## Rigid Monoidal Categories (has adjoints and CUPS and CAPS)
from discopy.rigid import Box, Id, Ty, Cap, Cup, Diagram
A = Ty('A')
print(A.l, 'is represented as', repr(A.l))
print(A.r, 'is represented as', repr(A.r))
assert A.r.l == A == A.l.r
# Equation(Cup(A.r, A.r.r), Cup(A, A.r), Cup(A.l, A), symbol='...').draw(figsize=(8, 1))
# Equation(Cap(A.l, A.l.l), Cap(A, A.l), Cap(A.r, A), symbol='...').draw(figsize=(8, 1))
# Snake Equations
snake1 = Id(A) @ Cap(A.r, A) >> Cup(A, A.r) @ Id(A)
snake2 = Cap(A, A.l) @ Id(A) >> Id(A) @ Cup(A.l, A)
assert snake1.normal_form() == Id(A) == snake2.normal_form() # normal form works for standard monoidals too
# print('Snake Equations - For any object', A, ':')
# Equation(snake1, Id(A), snake2).draw(figsize=(8, 2))
# Nested cups and caps
A, B = Ty('A'), Ty('B')
nested_cup = Diagram.cups(A @ B, (A @ B).r)
nested_cap = Diagram.caps((A @ B).r, A @ B)
nested_snake = Id(A @ B) @ nested_cap >> nested_cup @ Id(A @ B)
assert nested_snake.normal_form() == Id(A @ B)
Equation(nested_snake, nested_snake.normal_form()).draw()