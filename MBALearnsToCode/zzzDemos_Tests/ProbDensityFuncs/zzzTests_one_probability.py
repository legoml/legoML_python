from frozen_dict import FrozenDict as fdict
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import one_mass_function, one_density_function

o = one_mass_function(dict.fromkeys(('X', 'Y')), {fdict(X=0, Y=0), fdict(X=1, Y=2)}, {})

o_c = one_mass_function(dict.fromkeys(('X',)), {fdict(X=0), fdict(X=1)}, {'X': None}) #fdict(X=0): None