from __future__ import print_function
from copy import deepcopy
from pprint import pprint
import itertools
from sympy.integrals import integrate
from frozen_dict import FrozenDict
from MBALearnsToCode.Functions.FUNCTIONS___zzz_misc import combine_dict_and_kwargs, sympy_string_args
from MBALearnsToCode.UserDefinedClasses.CLASSES___DiscreteFunctions import DiscreteFiniteDomainFunction,\
    is_discrete_finite_domain_function
from MBALearnsToCode.Functions.FUNCTIONS___zzz_misc import dict_with_string_keys


class Factor():
    def __init__(self, sympy_function_or_discrete_finite_domain_function, conditions={}, scope=set()):
        self.conditions = dict_with_string_keys(conditions)
        if is_discrete_finite_domain_function(sympy_function_or_discrete_finite_domain_function):
            self.condition_instances = {}
            for string_args_and_values___frozen_dict in\
                    sympy_function_or_discrete_finite_domain_function.discrete_finite_mappings:
                condition_instance = {}
                for string_arg in (set(string_args_and_values___frozen_dict) & set(self.conditions)):
                    condition_instance[string_arg] = string_args_and_values___frozen_dict[string_arg]
                condition_instance = FrozenDict(condition_instance)
                self.condition_instances[string_args_and_values___frozen_dict] = condition_instance
            self.scope = sympy_function_or_discrete_finite_domain_function.args - set(self.conditions)
        else:
            self.scope = scope - set(self.conditions)
        self.function = sympy_function_or_discrete_finite_domain_function.copy()

    def copy(self):
        return Factor(self.function.copy(), deepcopy(self.conditions))

    def pprint(self):
        print('Factor Conditions:', self.conditions)
        print('Factor Scope:', self.scope)
        print('Factor Mappings:')
        if is_discrete_finite_domain_function(self.function):
            pprint(self.function.discrete_finite_mappings)
            print('   sum =', sum(self.function.discrete_finite_mappings.values()))
        else:
            print(self.function)

    def subs(self, args_and_values={}, **kw_args_and_values):
        args_and_values___dict = combine_dict_and_kwargs(args_and_values, kw_args_and_values)
        function = self.function.copy().subs(args_and_values___dict)
        return Factor(function, deepcopy(self.conditions))

    def max(self):  ### To refactor code
        condition_maxes = {}
        for args_and_values___frozen_dict, factor_value in self.function.discrete_finite_mappings.items():
            condition_instance = self.condition_instances[args_and_values___frozen_dict]
            if condition_instance in condition_maxes:
                condition_maxes[condition_instance] = max(condition_maxes[condition_instance], factor_value)
            else:
                condition_maxes[condition_instance] = factor_value
        d = {}
        for args_and_values___frozen_dict, factor_value in self.function.discrete_finite_mappings.items():
            if factor_value >= condition_maxes[self.condition_instances[args_and_values___frozen_dict]]:
                d[args_and_values___frozen_dict] = factor_value
        return Factor(DiscreteFiniteDomainFunction(d), deepcopy(self.conditions))

    def normalize(self, integration_ranges={}, **kw_integration_ranges):
        factor = self.copy()
        if is_discrete_finite_domain_function(factor.function):
            condition_sums = {}
            for string_args_and_values___frozen_dict, factor_value in factor.function.discrete_finite_mappings.items():
                condition_instance = factor.condition_instances[string_args_and_values___frozen_dict]
                if condition_instance in condition_sums:
                    condition_sums[condition_instance] += factor_value
                else:
                    condition_sums[condition_instance] = factor_value
            for string_args_and_values___frozen_dict in factor.function.discrete_finite_mappings:
                s = condition_sums[factor.condition_instances[string_args_and_values___frozen_dict]]
                if s > 0:
                    factor.function.discrete_finite_mappings[string_args_and_values___frozen_dict] /= s
        else:
            integration_ranges = combine_dict_and_kwargs(integration_ranges, kw_integration_ranges)
            integral = deepcopy(factor.function)
            for condition_arg in integration_ranges:
                lower_bound, upper_bound = integration_ranges[condition_arg]
                integral = integrate(integral, (condition_arg, lower_bound, upper_bound))
            factor.function /= integral
        return factor

    def eliminate(self, args_and_sum_and_values_or_integrate_and_range___tuple_or_list):
        function = self.function.copy()
        scope = deepcopy(self.scope)
        for arg, sum_or_integrate, values_or_bounds in args_and_sum_and_values_or_integrate_and_range___tuple_or_list:
            string_arg = str(arg)
            if sum_or_integrate == 'sum':
                values = values_or_bounds
                if is_discrete_finite_domain_function(function):
                    d = {}
                    for value in values:
                        d_subs = function.subs({string_arg: value}).discrete_finite_mappings
                        for string_args_and_values_0___frozen_dict, factor_value_0 in d_subs.items():
                            string_args_and_values_1___frozen_dict = FrozenDict(
                                set(string_args_and_values_0___frozen_dict.items()) - {(string_arg, value)})
                            if string_args_and_values_1___frozen_dict in d:
                                d[string_args_and_values_1___frozen_dict] += factor_value_0
                            else:
                                d[string_args_and_values_1___frozen_dict] = factor_value_0
                    function = DiscreteFiniteDomainFunction(d)
                else:
                    s = 0
                    for value in values:
                        s += function.subs({arg: value})
                    function = s
            elif sum_or_integrate == 'integrate':
                lower_bound, upper_bound = values_or_bounds
                if is_discrete_finite_domain_function(function):
                    d = function.discrete_finite_mappings
                    for string_args_and_values___frozen_dict, factor_value in d.items():
                        d[string_args_and_values___frozen_dict] = integrate(factor_value,
                                                                            (string_arg, lower_bound, upper_bound),
                                                                            manual=True)
                    function = DiscreteFiniteDomainFunction(d)
                else:
                    function = integrate(function, (string_arg, lower_bound, upper_bound), manual=True)
        return Factor(function, deepcopy(self.conditions), )

    def condition(self, eliminated_args_and_sum_and_values_or_integrate_and_range___tuple_or_list=None,
                  condition_args_and_values={}, **kw_condition_args_and_values):
        conditions = deepcopy(self.conditions)
        numerator_function = self.function.copy()
        if eliminated_args_and_sum_and_values_or_integrate_and_range___tuple_or_list:
            denominator_function = self.eliminate(
                eliminated_args_and_sum_and_values_or_integrate_and_range___tuple_or_list).function.copy()
            function = division_between_2_functions(numerator_function, denominator_function)
            for string_arg in denominator_function.args:
                conditions.update({string_arg: None})
        else:
            function = numerator_function.copy()
        condition_string_args_and_values___dict =\
            dict_with_string_keys(combine_dict_and_kwargs(condition_args_and_values, kw_condition_args_and_values))
        conditions.update(condition_string_args_and_values___dict)
        function = function.subs(condition_string_args_and_values___dict)
        if is_discrete_finite_domain_function(function):
            d = {}
            s0 = set(condition_string_args_and_values___dict.items())
            for string_args_and_values___frozen_dict, factor_value in function.discrete_finite_mappings.items():
                s = set(string_args_and_values___frozen_dict.items())
                if s >= s0:
                    d[FrozenDict(s - s0)] = factor_value
            function = DiscreteFiniteDomainFunction(d)
        return Factor(function, conditions)

    def multiply(self, *factors_to_multiply):
        conditions = deepcopy(self.conditions)   # just to be careful #
        scope = deepcopy(self.scope)   # just to be careful
        function = self.function.copy()
        for factor_to_multiply in factors_to_multiply:
            conditions.update(factor_to_multiply.conditions)
            scope.update(factor_to_multiply.scope)
            for arg in (set(conditions) & scope):
                conditions.pop(arg, None)
                scope.add(arg)
            function = product_of_2_functions(function, factor_to_multiply.function)
        return Factor(function, conditions)


def product_of_discrete_finite_domain_function_and_sympy_function(discrete_finite_domain_function, sympy_function):
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
            same_values_of_same_args &= (args_and_values_1___frozen_dict[arg] == args_and_values_2___frozen_dict[arg])
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
        return product_of_discrete_finite_domain_function_and_sympy_function(function_1, function_2)
    elif checks == (False, True):
        return product_of_discrete_finite_domain_function_and_sympy_function(function_2, function_1)
    else:
        return function_1 * function_2


def division_of_sympy_function_by_discrete_finite_domain_function(sympy_function, discrete_finite_domain_function):
    d = discrete_finite_domain_function.copy()
    for args_and_values___frozen_dict, function_value in d.discrete_finite_mappings.items():
        d.discrete_finite_mappings[args_and_values___frozen_dict] = sympy_function / function_value
    return d


def division_of_discrete_finite_domain_function_by_sympy_function(discrete_finite_domain_function, sympy_function):
    d = discrete_finite_domain_function.copy()
    for args_and_values___frozen_dict, function_value in d.discrete_finite_mappings.items():
        d.discrete_finite_mappings[args_and_values___frozen_dict] = function_value / sympy_function
    return d


def division_of_discrete_finite_domain_function_by_discrete_finite_domain_function(
        discrete_finite_domain_function_1, discrete_finite_domain_function_2):
    d = discrete_finite_domain_function_1.copy()
    for args_and_values___frozen_dict in d.discrete_finite_mappings:
        s = set(args_and_values___frozen_dict.items())
        for args_and_values_2___frozen_dict, function_value_2 in\
                discrete_finite_domain_function_2.discrete_finite_mappings.items():
            if s >= set(args_and_values_2___frozen_dict.items()):
                d.discrete_finite_mappings[args_and_values___frozen_dict] /= function_value_2
    return d


def division_between_2_functions(function_1, function_2):
    checks = (is_discrete_finite_domain_function(function_1), is_discrete_finite_domain_function(function_2))
    if checks == (True, True):
        return division_of_discrete_finite_domain_function_by_discrete_finite_domain_function(function_1, function_2)
    elif checks == (True, False):
        return division_of_discrete_finite_domain_function_by_sympy_function(function_1, function_2)
    elif checks == (False, True):
        return division_of_sympy_function_by_discrete_finite_domain_function(function_1, function_2)
    else:
        return function_1 / function_2