from copy import deepcopy
import itertools
from sympy import Symbol
from sympy.integrals import integrate
from frozen_dict import FrozenDict
from MachinePlayground.Functions.FUNCTIONS___zzz_misc import sympy_args
from MachinePlayground.UserDefinedClasses.CLASSES___functions import DiscreteFiniteDomainFunction


class Factor:
    def __init__(self, sympy_function_or_discrete_finite_domain_function, conditions={}):
        self.conditions = conditions
        if is_discrete_finite_domain_function(sympy_function_or_discrete_finite_domain_function):
            self.scope = sympy_function_or_discrete_finite_domain_function.args
        else:
            self.scope = sympy_args(sympy_function_or_discrete_finite_domain_function)
        self.function = sympy_function_or_discrete_finite_domain_function

    def copy(self):
        return Factor(self.function.copy(), deepcopy(self.conditions))

    def normalize(self, condition_scope_args_and_values_to_sum_over={},
                  **kw_condition_scope_args_and_values_to_sum_over):
        if condition_scope_args_and_values_to_sum_over:
            condition_scope_args_and_values_to_sum_over___dict =\
                deepcopy(condition_scope_args_and_values_to_sum_over)   # just to be careful #
            condition_scope_args_and_values_to_sum_over___dict.update(kw_condition_scope_args_and_values_to_sum_over)
        else:
            condition_scope_args_and_values_to_sum_over___dict = kw_condition_scope_args_and_values_to_sum_over
        factor = self.copy()
        denominator = 0
        s0 = set(condition_scope_args_and_values_to_sum_over___dict.items())
        for scope_args_and_values___frozen_dict in factor.function.discrete_finite_mappings:
            if set(scope_args_and_values___frozen_dict.items()) >= s0:
                denominator += factor.function.discrete_finite_mappings[scope_args_and_values___frozen_dict]
        for scope_args_and_values___frozen_dict in factor.function.discrete_finite_mappings:
            if set(scope_args_and_values___frozen_dict.items()) >= s0:
                factor.function.discrete_finite_mappings[scope_args_and_values___frozen_dict] /= denominator
        return factor

    def condition(self, condition_scope_args_and_values={}, **kw_condition_scope_args_and_values):
        if condition_scope_args_and_values:
            condition_scope_args_and_values___dict = deepcopy(condition_scope_args_and_values)   # just to be careful #
            condition_scope_args_and_values___dict.update(kw_condition_scope_args_and_values)
        else:
            condition_scope_args_and_values___dict = kw_condition_scope_args_and_values
        conditions = deepcopy(self.conditions)
        conditions.update(condition_scope_args_and_values___dict)
        function = self.function.copy().subs(condition_scope_args_and_values___dict)
        if is_discrete_finite_domain_function(function):
            d = {}
            s0 = set(condition_scope_args_and_values___dict.items())
            for scope_var_and_values___frozen_dict in function.discrete_finite_mappings:
                s = set(scope_var_and_values___frozen_dict.items())
                if s >= s0:
                    fd = FrozenDict(s - s0)
                    d[fd] = function.discrete_finite_mappings[scope_var_and_values___frozen_dict]
            function = DiscreteFiniteDomainFunction(d)
        return Factor(function, conditions)

    def eliminate(self, args_and_sum_and_values_or_integrate_and_range___tuple_or_list):
        conditions = deepcopy(self.conditions)   # just to be careful #
        function = self.function.copy()
        for arg, sum_or_integrate, values_or_bounds in args_and_sum_and_values_or_integrate_and_range___tuple_or_list:
            if sum_or_integrate == 'sum':
                values = values_or_bounds
                if is_discrete_finite_domain_function(function):
                    d = function.discrete_finite_mappings
                    d_subs = []
                    for value in values:
                        d_subs += [function.subs({arg: value}).discrete_finite_mappings]
                    d = {}
                    for d0 in d_subs:
                        for args_and_values_0__frozen_dict, function_value in d0.items():
                            args_and_values_1___frozen_dict = {}
                            for arg0 in args_and_values_0__frozen_dict:
                                if arg0 != arg:
                                    args_and_values_1___frozen_dict[arg0] = args_and_values_0__frozen_dict[arg0]
                            args_and_values_1___frozen_dict = FrozenDict(args_and_values_1___frozen_dict)
                            if args_and_values_1___frozen_dict in d:
                                d[args_and_values_1___frozen_dict] += function_value
                            else:
                                d[args_and_values_1___frozen_dict] = function_value
                    function = DiscreteFiniteDomainFunction(d)
                else:
                    s = 0
                    for value in values:
                        s += function.subs({str(arg): value})
                    function = s
            elif sum_or_integrate == 'integrate':
                lower_bound, upper_bound = values_or_bounds
                if is_discrete_finite_domain_function(function):
                    d = function.discrete_finite_mappings
                    for args_and_values___frozen_dict in d:
                        d[args_and_values___frozen_dict] = integrate(d[args_and_values___frozen_dict],
                                                                     (Symbol(str(arg)), lower_bound, upper_bound))
                    function = DiscreteFiniteDomainFunction(d)
                else:
                    function = integrate(function, (Symbol(str(arg)), lower_bound, upper_bound))
        return Factor(function, conditions)

    def multiply(self, *factors_to_multiply):
        conditions = deepcopy(self.conditions)   # just to be careful #
        f = self.function.copy()
        for factor_to_multiply in factors_to_multiply:
            conditions.update(factor_to_multiply.conditions)
            f = product_of_2_functions(f, factor_to_multiply.function)
        return Factor(f, conditions)


def is_discrete_finite_domain_function(function):
    return hasattr(function, 'discrete_finite_mappings')


def product_of_2_sympy_functions(sympy_function_1, sympy_function_2):
    return sympy_function_1 * sympy_function_2


def product_of_discrete_finite_domain_function_and_sumpy_function(sympy_function, discrete_finite_domain_function):
    d = discrete_finite_domain_function.copy()
    d.args.update(sympy_function.args)
    for args_and_values___frozen_dict, function_value in d.discrete_finite_mappings.items():
        d.discrete_finite_mappings[args_and_values___frozen_dict] = function_value * sympy_function
    return d


def product_of_2_discrete_finite_domain_functions(discrete_finite_domain_function_1, discrete_finite_domain_function_2):
    d = DiscreteFiniteDomainFunction({})
    d.args = discrete_finite_domain_function_1.args.union(discrete_finite_domain_function_2.args)
    d1 = discrete_finite_domain_function_1.discrete_finite_mappings
    d2 = discrete_finite_domain_function_2.discrete_finite_mappings
    for d1_item, d2_item in itertools.product(d1.items(), d2.items()):
        args_and_values_1___frozen_dict, function_value_1 = d1_item
        args_and_values_2___frozen_dict, function_value_2 = d2_item
        same_values_of_same_args = True
        for arg in (set(args_and_values_1___frozen_dict) & set(args_and_values_2___frozen_dict)):
            if args_and_values_1___frozen_dict[arg] != args_and_values_2___frozen_dict[arg]:
                same_values_of_same_args = False
        if same_values_of_same_args:
            d.discrete_finite_mappings[FrozenDict(set(args_and_values_1___frozen_dict.items()) |
                                                  set(args_and_values_2___frozen_dict.items()))] =\
                function_value_1 * function_value_2
    return d


def product_of_2_functions(function_1, function_2):
    checks = (is_discrete_finite_domain_function(function_1), is_discrete_finite_domain_function(function_2))
    if checks == (True, True):
        return product_of_2_discrete_finite_domain_functions(function_1, function_2)
    elif checks == (True, False):
        return product_of_discrete_finite_domain_function_and_sumpy_function(function_1, function_2)
    elif checks == (False, True):
        return product_of_discrete_finite_domain_function_and_sumpy_function(function_2, function_1)
    else:
        return product_of_2_sympy_functions(function_1, function_2)