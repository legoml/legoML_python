from copy import deepcopy


class Factor: 
    def __init__(self, scope_vars_values_and_factor_values___dict):
        scope_vars_values_and_factor_values___dict =\
            deepcopy(scope_vars_values_and_factor_values___dict)   # just to be careful
        scope = {}
        for vars_and_values in scope_vars_values_and_factor_values___dict:
            for var, value in vars_and_values:
                if var in scope:
                    scope[var].add(value)
                else:
                    scope[var] = {value}
        self.scope = scope
        self.mappings = scope_vars_values_and_factor_values___dict