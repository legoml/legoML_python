from copy import copy as shallow_copy, deepcopy
from MBALearnsToCode.Functions.FUNCTIONS___zzzMISC import combine_dict_and_kwargs, sympy_string_args, sympy_subs


class DiscreteFiniteDomainFunction:
    def __init__(self, discrete_finite_mappings={}, **kw_discrete_finite_mappings):
        args_values_and_function_values___dict = combine_dict_and_kwargs(discrete_finite_mappings,
                                                                         kw_discrete_finite_mappings)
        self.args = set()
        for args_and_values___frozen_dict, function_value in args_values_and_function_values___dict.items():
            self.args.update(args_and_values___frozen_dict.keys())
            self.args.update(sympy_string_args(function_value))
        self.discrete_finite_mappings = args_values_and_function_values___dict

    def copy(self):
        return DiscreteFiniteDomainFunction(shallow_copy(self.discrete_finite_mappings))

    def subs(self, args_and_values={}, **kw_args_and_values):
        args_and_values___dict = combine_dict_and_kwargs(args_and_values, kw_args_and_values)
        s0 = set(args_and_values___dict.items())
        d = {}
        for args_and_values___frozen_dict, function_value in self.discrete_finite_mappings.items():
            s1 = s0 - set(args_and_values___frozen_dict.items())
            s = set()
            for arg, value in s1:
                s.update(str(arg))
            if s <= sympy_string_args(function_value):
                d[args_and_values___frozen_dict] = sympy_subs(function_value, args_and_values___dict)
        return DiscreteFiniteDomainFunction(d)