from discopy.grammar.pregroup import Cap, Cup, Id, Ty, Word
from discopy.drawing import Equation
from discopy import monoidal
from lambeq import pregroups, AtomicType, RewriteRule, Rewriter, BobcatParser, create_pregroup_diagram, remove_cups, IQPAnsatz, DepCCGParser
import matplotlib.font_manager as fm # allows display of kana and kanji
import matplotlib.pyplot as plt
import random
fprop = fm.FontProperties(fname='./Paper3Code/fonts/NotoSansJP-VariableFont_wght.ttf') # set font

# # Prepare some data to test font
# x = list(range(20))
# xticks = ["類別{:d}".format(i) for i in x]
# y = [random.randint(10,99) for i in x]

# # Plot the graph
# plt.figure(figsize=(8, 2))
# plt.bar(x, y)
# plt.xticks(x, xticks, fontproperties=fprop, fontsize=12, rotation=45)
# plt.title("圖1", fontproperties=fprop, fontsize=18)
# plt.show()

## Reference extending_lambeq.py, discocat_test.py, training in the hybrid case, and the quantum case
## NOTE: Creating a custom enum is problematic in Python
## class HonorificType(AtomicType): will likely not work and will need to use Ty() instead

## TODO: Will need to read in data for ML tasks. I'll need a corpus split into train, test, and dev.
# The corpus should contain casual and polite sentences with a "1" as the label for polite ones
# May need a Japanese corpus and an English corpus. The ML task would be "selecting" rather than
# translating at that point.

def use_complex_types():
    ## Define pregroup types (NOTE: starting with only 2 levels of politeness for an attainable scope)
    # h is polite. The absence of h means casual when there are only 2 levels.
    # Formal words add an h.l that anticipates the formal copula, desu, or masu.
    n, s, h = Ty('n'), Ty('s'), Ty('h')
    nom, gen, dat, acc, loc_inst = Ty('nom'), Ty('gen'), Ty('dat'), Ty('acc'), Ty('loc_inst')
    ab, term = Ty('ab'), Ty('term')

    ## Define common complex Word types
    # alternatively, particle types can be sucked into noun phrases and, thus, ignored at a cost.
    # There are a lot of times when Japanese drops words and the meaning is inferred. Computers cannot do this
    # so, I propose adding the dropped portions back in.
    verb_connective_ga = nom.r @ s @ h.l         # used when attaching masu to the verb stem of a simple subject sentence
    verb_connective_ga_ni = nom.r @ dat.r @ s @ h.l
    verb_connective_n_phrase = n.r @ s @ h.l     # used when grouping particles into a noun phrase
    masu = h
    verb_casual_ga = nom.r @ s
    verb_casual_n_phrase = n.r @ s               # used when grouping particles into a noun phrase
    da = n.r @ s
    desu = h @ s                    # This is the case where desu is the copula AND adding politeness
    desu_i_adj = masu               # in this case, it does nothing other than add the h type
    pred_polite_n = n @ h.l         # predicate noun's in a polite sentence anticipate desu
    adj_na = n                      # a na-adjective is simply treated as a noun in all cases
    pro = n                         # a pronoun is treated as a noun as well. This could be expanded with h types.
    adj_i = verb_casual_ga          # an i-adjective functions like a casual verb

    ## Define special words: We may want to intentionally ignore or target these with rewrite rules.
    # Some of these definitions are just placeholders and may not become important for the present study.
    # A "None" means I am uncertain how to type the word for now.
    particles_logical = {'ga': n.r @ nom, 'de': n.r @ loc_inst, 'no': n.r @ gen, 
                        'ni': n.r @ dat, 'he': n.r @ dat, 'wo': n.r @ acc, 'kara': n.r @ ab, 
                        'made': n.r @ term, 'yori': n.r @ ab @ h.l}
    particles_conjunctions = {'to': None, 'ya': None, 'toka': None}
    particles_topic_markers = {'ha': n.r @ s @ s.l, 'mo': n.r @ s @ s.l}
    particles_sentence_ending = {'ka': None , 'ne': None, 'ze': None, 'yo': None, 'zo': None}
    # Formality is more granular in actuality, but this suffices for 2 levels.
    # These titles do not account for the different shades of formality between them.
    titles = {'san': n.r @ h, 'sama': n.r @ h, 'chan': n.r @ h, 'kun': n.r @ h} 

    ## The stem of the verb is what carries the meaning of the sentence in a non-topical sentence.
    ## The copula or polite verb is going to be strictly treated as a politeness marker for now.
    ## This is supported by the fact that the the casual version of a verb is used to modify succeeding nouns.
    ## There is also a school of thought that suggests combining noun phrases into a single unit.
    ## In the Japanese context, this would allow us to treat 'kanojo ga' differently from 'kanojo de'
    ## without having to specifically handle the particle.
    # Simple drawing example
    ga_words = [Word('kanojo', n), Word('ga', particles_logical['ga']), Word('kaeri', verb_connective_ga), 
                Word('masu', masu)]
    cups_ga_verb_masu = Cup(n, n.r) @ Cup(nom, nom.r) @ Id(s) @ Cup(h.l, h)
    # remember that diagrams are drawn top-down with '>>' or '<<' dividing the layers.
    ga_diagram = Id().tensor(*ga_words) >> cups_ga_verb_masu
    # pregroups.draw(ga_diagram)
    # monoidal.Diagram.draw(remove_cups(ga_diagram))

    # Complex drawing example. This approach can be used to build cups without needing to define them
    # For complex examples, look for the particles in the sentence then make the sentence into a normal
    # form based on typical word order in practice. This will allow standardization of indices for cups.
    ga_ni_words = Word('kanojo', n) @ Word('ga', particles_logical['ga']) @ Word('uchi', n) @ Word('ni', particles_logical['ni']) @ Word('kaeri', verb_connective_ga_ni) @ Word('masu', masu)
    raw_ga_ni_diagram = ga_ni_words.cup(2, 6).cup(4, 5).cup(5, 6).cup(2, 3).cup(0, 1).draw() 
    ga_ni_diagram = remove_cups(raw_ga_ni_diagram) # removes post selections to speed up quantum experiment
    ga_ni_diagram_nf = ga_ni_diagram.normal_form() # stretches/shrinks wires to make diagram more compact
    # monoidal.Diagram.draw(ga_ni_diagram)
    # monoidal.Diagram.draw(ga_ni_diagram_nf)

    ## Make circuits and set Ansatz
    ansatz = IQPAnsatz({n: 1, s: 1, h: 1, nom: 1, dat: 1, loc_inst : 1, gen: 1, acc: 1, ab: 1, term: 1},
                    n_layers=1, n_single_qubit_params=3)
    ga_circuit = ansatz(remove_cups(ga_diagram))
    # ga_circuit.draw(figsize=(6, 8)) # This format of circuit drawing follows the shape of the monoidal diagram
    ga_ni_circuit = ansatz(ga_ni_diagram_nf)
    # ga_ni_circuit.draw(figsize=(6, 8))

    ## TODO: Train Model on circuits
    # I'll need a corpus of sentences to translate into diagrams first...

#--------------------------------------------------------------------------------------
## For now, these types are more viable experimentally. More types means more quantum operations
## that have to be implemented. Adding more than one is probably not the pest way to proceed for now.
def use_simple_types():
    # define types
    n, s, h = Ty('n'), Ty('s'), Ty('h')
     
    # define common grammar connections. Reference complex types above for differences
    verb_connective_single = s.l @ s @ h.l         # used when attaching masu to the verb stem of a simple subject sentence
    verb_connective_double = s.l @ s.l @ s @ h.l
    verb_connective_triple = s.l @ s.l @ s.l @ s @ h.l
    masu = h
    desu_i_adj = masu               # in this case, it does nothing other than add the h type
    verb_casual_single = s.l @ s
    verb_casual_double = s.l @ s.l @ s
    verb_casual_triple = s.l @ s.l @ s.l @ s
    adj_i_s = verb_casual_single    # an i-adjective functions like a casual verb
    adj_i = n @ n.l
    adj_na = adj_i                  # will be used with "na" suffixed as one word. Otherwise, it is just a noun.
    da = n.r @ s
    desu_n = h @ s                  # This is the case where desu is the copula AND adding politeness
    pred_polite_n = n @ h.l         # predicate noun's in a polite sentence anticipate desu
    pro = n                         # a pronoun is treated as a noun as well. This could be expanded with h types.
    
    #'no': n.r @ n.l
    # Need to double check the .ll idea. I'd like to use it as that is how lambeq handles some of this situations.
    # you could just have several s types and cancel out all but one.
    particles_logical = n.r @ s.ll 
    particles_topic_markers =  n.r @ s @ s.ll
    
    # titles have subtle levels of politeness associated with them that merit a more in depth study
    titles = {'san': n.r @ h, 'sama': n.r @ h, 'chan': n.r @ h, 'kun': n.r @ h}

#--------------------------------------------------------------------------------------
## TODO: Write Rewriters, Functors, etc. below. Likely will put them in another file.
## Classes (NOTE: Past tense does not require a change to the grammatical links in the quantum network.)
## Ideas for other RewriteRules: Negation, To Honorific, To Casual, Combine particle with noun, 
##                               Convert from non-topical to topical ('ga' to 'ha')
class PastRewriteRule(RewriteRule):
    # The verb endings will need to be tokenized off of the stems
    mapping ={ 'desu': 'deshita', 'masu': 'mashita', 'ru': 'tta', 'i': 'katta' }
    def matches(self, box):
        return box.name in self.mapping
    def rewrite(self, box):
        new_name = self.mapping[box.name]
        return type(box)(name=new_name, dom=box.dom, cod=box.cod)
    
## Procedural section begins 
use_complex_types()
parser = BobcatParser(verbose='text')  #BobcatParser does not work on Japanese, so I'll need to work on simple types and manually parse like I did in complex types.
d = parser.sentence2diagram('sakana ha shinsen desu')
#jp_d = parser.sentence2diagram('これ は 新鮮 です') #Japanese isn't shown on the diagrams even if it will show on matplotlib
past_d = Rewriter([PastRewriteRule()])(d)
Equation(d, past_d, symbol='->').draw(figsize=(14, 4))

## tried depccg parser for Japanese, but it will not install because it doesn't see Cython dependency.
# dep_parser = DepCCGParser(lang='jp')
# dep_d = dep_parser.sentence2diagram('sakana ha shinsen desu')
# Equation(dep_d, d, symbol='->').draw(figsize=(14, 4)) 