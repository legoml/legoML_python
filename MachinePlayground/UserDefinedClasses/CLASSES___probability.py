from copy import deepcopy
import itertools


class Factor: 
    def __init__(self, scope_vars_values_and_factor_values___dict):
        self.scope = {}
        self.mappings = {}
        for scope_vars_and_values, factor_value in scope_vars_values_and_factor_values___dict.items():
            for var, value in scope_vars_and_values:
                if var in self.scope:
                    self.scope[var].add(value)
                else:
                    self.scope[var] = {value}
            self.mappings[frozenset(scope_vars_and_values)] = factor_value

    def normalize(self):
        factor = deepcopy(self)
        s = sum(factor.mappings.values())
        for scope_vars_and_values, factor_value in factor.mappings.items():
            factor.mappings[scope_vars_and_values] = factor_value / s
        return factor

    def condition(self, scope_vars_and_values___dict):
        d = {}
        s = set(scope_vars_and_values___dict.items())
        for scope_var_and_values, factor_value in self.mappings.items():
            if scope_var_and_values >= s:
                d[scope_var_and_values - s] = factor_value
        return Factor(d)

    def margin(self, scope_vars_to_sum_over):
        factor = Factor({})
        factor.scope = deepcopy(self.scope)   # just to be careful #
        for var in scope_vars_to_sum_over:
            del factor.scope[var]
        for scope_vars_and_values, factor_value in self.mappings.items():
            scope_vars_remaining = set()
            for var, value in scope_vars_and_values:
                if var not in scope_vars_to_sum_over:
                    scope_vars_remaining.add((var, value))
            scope_vars_remaining = frozenset(scope_vars_remaining)
            if scope_vars_remaining in factor.mappings:
                factor.mappings[scope_vars_remaining] += factor_value
            else:
                factor.mappings[scope_vars_remaining] = factor_value
        return factor

    def product(self, *factors_to_multiply):
        factor = self
        empty_factor = Factor({})
        for factor_to_multiply in factors_to_multiply:
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
                same_values_of_same_vars = True
                for var in (set(vars_and_values_0___dict) & set(vars_and_values_1___dict)):
                    if vars_and_values_0___dict[var] != vars_and_values_1___dict[var]:
                        same_values_of_same_vars = False
                if same_values_of_same_vars:
                    f.mappings[frozenset(vars_and_values_0) | frozenset(vars_and_values_1)] =\
                        factor_value_0 * factor_value_1
            factor = f
        return deepcopy(factor)

