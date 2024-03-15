from lambeq.backend.grammar import Cup, Diagram, Id, Swap, Ty, Word
from lambeq.backend.drawing import draw, draw_equation
from lambeq import IQPAnsatz, RemoveCupsRewriter

## define the grammatical types
# note the "h" type for honorifics/polite form
n, s, h = Ty('n'), Ty('s'), Ty ('h') 

## define utilities
remove_cups = RemoveCupsRewriter()
ansatz = IQPAnsatz({n: 1, s: 1, h: 1}, n_layers=1, n_single_qubit_params=3)

## define common, simple grammar connections.
# polite
pred_polite_n = n @ h.l
verb_connective_single = s.l @ s @ h.l     
verb_connective_double = s.l @ s.l @ s @ h.l
verb_connective_triple = s.l @ s.l @ s.l @ s @ h.l
masu = h
desu_i_adj = masu
desu_n = h @ s                
# casual
verb_casual_single = s.l @ s
verb_casual_double = s.l @ s.l @ s
verb_casual_triple = s.l @ s.l @ s.l @ s
adj_i_s = verb_casual_single   
adj_i = n @ n.l
adj_na = adj_i                 
da = n.r @ s                    
pro = n
# particles 
particle_logical = n.r @ s.l.l

### Polite sentence examples
## example self-move sentence. 
# Define words. Define grammatical links.
words = [Word('kanojo', n), Word('ga', particle_logical), Word('kaeri', verb_connective_single), 
                Word('masu', masu)]
cups_single_verb_masu = Cup(n, n.r) @ Cup(s.l.l, s.l) @ Id(s) @ Cup(h.l, h)
single_diagram = Id().tensor(*words) >> cups_single_verb_masu
single_diagram = single_diagram.normal_form()
draw(single_diagram)
# equivalent example using cups and swaps method.
morphisms = [(Cup, 0, 1), (Cup, 2, 3), (Cup, 5, 6)]
diagram = Diagram.create_pregroup_diagram(words, morphisms)
diagram = diagram.normal_form()
draw(diagram)
# demonstrate equivalency with an equation
draw_equation(single_diagram, diagram, symbol='==', figsize=(10, 4), draw_as_pregroup=False, foliated=True)
# simplify diagram by removing Cups
simplified_diagram = remove_cups(diagram)
simplified_diagram.draw(figsize=(9, 10))
# turn simplified diagram into quantum circuit
circuit = ansatz(simplified_diagram)
circuit.draw(figsize=(9,10)) 

# ## example other-move sentence. Follows the same steps as above.
# # cups and swaps method. You cannot cross the wires! Verbose method is too cumbersome with longer sentences.
om_words = [Word('kanojo', n), Word('ga', particle_logical), Word('kami', n), 
            Word('wo', particle_logical), Word('ori', verb_connective_double), Word('masu', masu)]
om_morphisms = [(Cup, 0, 1), (Cup, 3, 4), (Cup, 5, 6), (Cup, 9, 10), (Cup, 2, 7)] # connect adjacents first!
om_diagram = Diagram.create_pregroup_diagram(om_words, om_morphisms)
om_diagram = om_diagram.normal_form()
draw(om_diagram)
simplified_om_diagram = (remove_cups(om_diagram))
om_circuit = ansatz(simplified_om_diagram)
om_circuit.draw(figsize=(9,10)) 

## example triple particle sentence: other-move with "ni" or "de"
de_words = [Word('kanojo', n), Word('ga', particle_logical), Word('tsukue', n), Word('de', particle_logical), 
            Word('kami', n), Word('wo', particle_logical), Word('ori', verb_connective_triple), Word('masu', masu)]
de_morphisms = [(Cup, 0, 1), (Cup, 3, 4), (Cup, 6, 7), (Cup, 8, 9), (Cup, 13, 14), (Cup, 5, 10), (Cup, 2, 11)]
de_diagram = Diagram.create_pregroup_diagram(de_words, de_morphisms)
de_diagram = de_diagram.normal_form()
draw(de_diagram)
simplified_de_diagram = (remove_cups(de_diagram))
de_circuit = ansatz(simplified_de_diagram)
de_circuit.draw(figsize=(9,10)) 

### Casual sentence examples
## example self-move sentence
cs_words = [Word('kanojo', n), Word('ga', particle_logical), Word('kaeru', verb_casual_single)]
cs_morphisms = [(Cup, 0, 1), (Cup, 2, 3)]
cs_diagram = Diagram.create_pregroup_diagram(cs_words, cs_morphisms)
cs_diagram = cs_diagram.normal_form()
draw(cs_diagram)
simplified_cs_diagram = (remove_cups(cs_diagram))
cs_circuit = ansatz(simplified_cs_diagram)
cs_circuit.draw(figsize=(9,10)) 

## example other-move sentence
co_words = [Word('kanojo', n), Word('ga', particle_logical), Word('kami', n), 
             Word('wo', particle_logical), Word('oru', verb_casual_double)]
co_morphisms = [(Cup, 0, 1), (Cup, 3, 4), (Cup, 5, 6), (Cup, 2, 7)]
co_diagram = Diagram.create_pregroup_diagram(co_words, co_morphisms)
co_diagram = co_diagram.normal_form()
draw(co_diagram)
simplified_co_diagram = (remove_cups(co_diagram))
co_circuit = ansatz(simplified_co_diagram)
co_circuit.draw(figsize=(9,10)) 

## example triple particle sentence: other-move with "ni" or "de"
cde_words = [Word('kanojo', n), Word('ga', particle_logical), Word('tsukue', n), Word('de', particle_logical), 
             Word('kami', n), Word('wo', particle_logical), Word('oru', verb_casual_triple)]
cde_morphisms = [(Cup, 0, 1), (Cup, 3, 4), (Cup, 6, 7), (Cup, 8, 9), (Cup, 5, 10), (Cup, 2, 11)]
cde_diagram = Diagram.create_pregroup_diagram(cde_words, cde_morphisms)
cde_diagram = cde_diagram.normal_form()
draw(cde_diagram)
simplified_cde_diagram = (remove_cups(cde_diagram))
cde_circuit = ansatz(simplified_cde_diagram)
cde_circuit.draw(figsize=(9,10)) 

## Next steps: 
# 2. Create a training set and a validation set
# 3. Determine a generic way to read in sentences from the set and build diagrams for them
#       properties: cup_count = word_count - 1; morphisms connect adjacent then connect non-adjacent
#                   left to right with the largest span; word_count == token_count;
# 4. Train a model to identify honorific v. non-honorific sentences from the sets or similar