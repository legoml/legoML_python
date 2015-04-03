from copy import deepcopy
import itertools
from MachinePlayground.UserDefinedClasses.CLASSES___probability import Factor


def factor_product(*factors):
    factor = factors[0]
    empty_factor = Factor({})
    for factor_to_multiply in factors[1:]:
        f = deepcopy(empty_factor)   # just to be careful #
        f.scope = deepcopy(factor.scope)   # just to be careful #
        for var in factor_to_multiply.scope:
            if var in f.scope:
                f.scope[var].update(factor_to_multiply.scope[var])
            else:
                f.scope[var] = factor_to_multiply.scope[var]
        d0 = factor.mappings
        d1 = factor_to_multiply.mappings
        for d0_item, d1_item in itertools.product(d0.items(), d1.items()):
            vars_and_values_0, factor_value_0 = d0_item
            vars_and_values_1, factor_value_1 = d1_item
            vars_and_values_0___dict = dict(vars_and_values_0)
            vars_and_values_1___dict = dict(vars_and_values_1)
            sames_values_of_same_vars = True
            for var in set(vars_and_values_0___dict).intersection(set(vars_and_values_1___dict)):
                if vars_and_values_0___dict[var] != vars_and_values_1___dict[var]:
                    sames_values_of_same_vars = False
            if sames_values_of_same_vars:
                f.mappings[tuple(set(vars_and_values_0).union(set(vars_and_values_1)))] =\
                    factor_value_0 * factor_value_1
        factor = f
    return deepcopy(factor)   # just to be careful #