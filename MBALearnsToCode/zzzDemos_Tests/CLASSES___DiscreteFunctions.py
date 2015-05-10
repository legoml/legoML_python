from MBALearnsToCode.Functions.FUNCTIONS___zzz_misc import combine_dict_and_kwargs, dict_with_string_keys,\
    frozen_dict_with_string_keys, sympy_string_args, sympy_subs


class DiscreteFiniteDomainFunction:
    def __init__(self, discrete_finite_mappings={}, **kw_discrete_finite_mappings):
        args_values_and_function_values___dict = combine_dict_and_kwargs(discrete_finite_mappings,
                                                                         kw_discrete_finite_mappings)
        self.args = set()
        self.discrete_finite_mappings = {}
        for args_and_values___frozen_dict, function_value in args_values_and_function_values___dict.items():
            string_args_and_values___frozen_dict = frozen_dict_with_string_keys(args_and_values___frozen_dict)
            self.args.update(string_args_and_values___frozen_dict.keys())
            self.args.update(sympy_string_args(function_value))
            self.discrete_finite_mappings[string_args_and_values___frozen_dict] = function_value

    def copy(self):
        return DiscreteFiniteDomainFunction(self.discrete_finite_mappings.copy())

    def subs(self, subs_args_and_values={}, **kw_subs_args_and_values):
        subs_string_args_and_values___dict = dict_with_string_keys(combine_dict_and_kwargs(subs_args_and_values,
                                                                                           kw_subs_args_and_values))
        s0 = set(subs_string_args_and_values___dict.items())
        d = {}
        for string_args_and_values___frozen_dict, function_value in self.discrete_finite_mappings.items():
            spare_string_args = set(dict(s0 - set(string_args_and_values___frozen_dict.items())).keys())
            if spare_string_args <= sympy_string_args(function_value):
                d[string_args_and_values___frozen_dict] = sympy_subs(function_value, subs_string_args_and_values___dict)
        return DiscreteFiniteDomainFunction(d)


def is_discrete_finite_domain_function(function):
    return hasattr(function, 'discrete_finite_mappings')



