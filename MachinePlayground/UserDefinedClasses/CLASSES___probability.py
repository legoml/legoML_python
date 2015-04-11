from copy import deepcopy
import itertools
from frozen_dict import FrozenDict


class Factor: 
    def __init__(self, scope_vars_values_and_factor_values___dict):
        self.scope = set()
        for scope_vars_and_values___frozen_dict in scope_vars_values_and_factor_values___dict:
            self.scope.update(scope_vars_and_values___frozen_dict.keys())
        self.mappings = scope_vars_values_and_factor_values___dict.copy()   # shallow copy, hopefully okay #

    def copy(self):
        return Factor(self.mappings)

    def normalize(self):
        factor = self.copy()
        s = sum(factor.mappings.values())
        for scope_vars_and_values___frozen_dict in factor.mappings:
            factor.mappings[scope_vars_and_values___frozen_dict] /= s   # MUST NOT do in-place change #
        return factor

    def condition(self, condition_scope_vars_and_values___dict):
        factor = Factor({})
        s0 = set(condition_scope_vars_and_values___dict.items())
        for scope_var_and_values___frozen_dict in self.mappings:
            s = set(scope_var_and_values___frozen_dict.items())
            if s >= s0:
                d = FrozenDict(s - s0)
                factor.scope.update(d.keys())
                factor.mappings[d] = self.mappings[scope_var_and_values___frozen_dict]
        return factor

    def margin(self, scope_vars_to_sum_over):
        factor = Factor({})
        factor.scope = deepcopy(self.scope)   # just to be careful #
        for var in scope_vars_to_sum_over:
            factor.scope.remove(var)
        for scope_vars_and_values___frozen_dict in self.mappings:
            scope_vars_remaining = set()
            for var in scope_vars_and_values___frozen_dict:
                if var not in scope_vars_to_sum_over:
                    scope_vars_remaining.add((var, scope_vars_and_values___frozen_dict[var]))
            scope_vars_remaining = FrozenDict(scope_vars_remaining)
            if scope_vars_remaining in factor.mappings:
                factor.mappings[scope_vars_remaining] += self.mappings[scope_vars_and_values___frozen_dict]
            else:
                factor.mappings[scope_vars_remaining] = self.mappings[scope_vars_and_values___frozen_dict]
        return factor

    def product(self, *factors_to_multiply):
        factor = self.copy()
        for factor_to_multiply in factors_to_multiply:
            f = Factor({})
            f.scope = deepcopy(factor.scope)   # just to be careful #
            for var in factor_to_multiply.scope:
                f.scope.add(var)
            d0 = factor.mappings
            d1 = factor_to_multiply.mappings
            for d0_item, d1_item in itertools.product(d0.items(), d1.items()):
                vars_and_values_0___frozen_dict, factor_value_0 = d0_item
                vars_and_values_1___frozen_dict, factor_value_1 = d1_item
                same_values_of_same_vars = True
                for var in (set(vars_and_values_0___frozen_dict) & set(vars_and_values_1___frozen_dict)):
                    if vars_and_values_0___frozen_dict[var] != vars_and_values_1___frozen_dict[var]:
                        same_values_of_same_vars = False
                if same_values_of_same_vars:
                    f.mappings[FrozenDict(set(vars_and_values_0___frozen_dict.items()) |
                                          set(vars_and_values_1___frozen_dict.items()))] =\
                        factor_value_0 * factor_value_1
            factor = f
        return factor