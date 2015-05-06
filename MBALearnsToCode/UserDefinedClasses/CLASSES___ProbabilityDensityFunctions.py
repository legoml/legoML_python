from copy import deepcopy
import itertools
from sympy import exp, log, pi
from sympy.matrices import BlockMatrix, det
from frozen_dict import FrozenDict
from MBALearnsToCode.Functions.FUNCTIONS___sympy import sympy_xreplace_doit_explicit_evalf, is_sympy_expression


class ProbabilityDensityFunction:
    def __init__(self, family, var_symbols, parameters, density_lambda, normalization_lambda, max_lambda,
                 marginalization_lambda, conditioning_lambda, conditions={}):
        self.family = family
        self.vars = var_symbols
        self.parameters = parameters
        self.conditions = conditions
        self.scope = set(var_symbols) - set(conditions)
        self.density_lambda = density_lambda
        self.normalization_lambda = normalization_lambda
        self.max_lambda = max_lambda
        self.marginalization_lambda = marginalization_lambda
        self.conditioning_lambda = conditioning_lambda

    def copy(self):
        return ProbabilityDensityFunction(self.family, deepcopy(self.vars), self.parameters.copy(),
            self.density_lambda, self.normalization_lambda, self.max_lambda, self.marginalization_lambda,
            self.conditioning_lambda, deepcopy(self.conditions))

    def __call__(self, vars_and_parameters_values___dict={}, return_probability=True):
        if vars_and_parameters_values___dict:
            symbols_and_values___dict = {}
            for var_or_parameter, value in vars_and_parameters_values___dict.items():
                if var_or_parameter in self.vars:
                    symbols_and_values___dict[self.vars[var_or_parameter]] = value
                elif var_or_parameter in self.parameters:
                    symbols_and_values___dict[self.parameters[var_or_parameter]] = value
            if return_probability:
                return sympy_xreplace_doit_explicit_evalf(
                    return_probability_from_minus_log_probability(self.density_lambda(self.vars, self.parameters)),
                    symbols_and_values___dict)
            else:
                return sympy_xreplace_doit_explicit_evalf(self.density_lambda(self.vars, self.parameters),
                                                          symbols_and_values___dict)
        elif return_probability:
            return return_probability_from_minus_log_probability(self.density_lambda(self.vars, self.parameters))
        else:
            return self.density_lambda(self.vars, self.parameters)

    def normalize(self):
        return self.normalization_lambda(self)

    def max(self):
        return self.max_lambda(self)

    def marginalize(self, marginalized_vars):
        return self.marginalization_lambda(self, marginalized_vars)

    def condition(self, conditions={}):
        return self.conditioning_lambda(self, conditions)

    def multiply(self, *probability_density_functions_to_multiply):
        pdf = self.copy()
        for pdf_to_multiply in probability_density_functions_to_multiply:
            pdf = product_of_2_probability_density_functions(pdf, pdf_to_multiply)
        return pdf


def return_probability_from_minus_log_probability(sympy_expression):
    if isinstance(sympy_expression, dict):
        return {k: exp(-v) for k, v in sympy_expression.items()}
    else:
        return exp(-sympy_expression)


def product_of_2_probability_density_functions(pdf_1, pdf_2):
    families = (pdf_1.family, pdf_2.family)
    if families == ('DiscreteFinite', 'DiscreteFinite'):
        return product_of_2_discrete_finite_probability_mass_functions(pdf_1, pdf_2)
    elif pdf_1.family == 'DiscreteFinite':
        return product_of_discrete_finite_probability_mass_function_and_continuous_probability_density_function(
            pdf_1, pdf_2)
    elif pdf_2.family == 'DiscreteFinite':
        return product_of_discrete_finite_probability_mass_function_and_continuous_probability_density_function(
            pdf_2, pdf_1)
    elif families == ('Gaussian', 'Gaussian'):
        return product_of_2_gaussian_probability_density_functions(pdf_1, pdf_2)
    else:
        return None


def discrete_finite_mass(var_values___dict, parameters):
    var_point_values___dict = deepcopy(var_values___dict)
    for var, value in var_values___dict.items():
        if (value is None) or is_sympy_expression(value):
            del var_point_values___dict[var]
    s0 = set(var_point_values___dict.items())
    d = {}
    mappings = parameters['mappings']
    for var_values___frozen_dict, function_value in mappings.items():
        spare_var_values = dict(s0 - set(var_values___frozen_dict.items()))
        s = set(spare_var_values.keys())
        if not(s) or (s and not(s & set(var_values___frozen_dict))):
            d[var_values___frozen_dict] = sympy_xreplace_doit_explicit_evalf(function_value, var_values___dict)
    return d


def discrete_finite_normalization(discrete_finite_pmf):
    pmf = discrete_finite_pmf.copy()
    mappings = pmf.parameters['mappings']
    condition_instances = pmf.parameters['condition_instances']
    condition_sums = {}
    for var_values___frozen_dict, function_value in mappings.items():
        condition_instance = condition_instances[var_values___frozen_dict]
        if condition_instance in condition_sums:
            condition_sums[condition_instance] += function_value
        else:
            condition_sums[condition_instance] = function_value
    for var_values___frozen_dict in mappings:
        s = condition_sums[condition_instances[var_values___frozen_dict]]
        if s > 0:
            pmf.parameters['mappings'][var_values___frozen_dict] /= s
    return pmf


def discrete_finite_max(discrete_finite_pmf):
    mappings = discrete_finite_pmf.parameters['mappings']
    condition_instances = discrete_finite_pmf.parameters['condition_instances']
    condition_maxes = {}
    for var_values___frozen_dict, function_value in mappings.items():
        condition_instance = condition_instances[var_values___frozen_dict]
        if condition_instance in condition_maxes:
            condition_maxes[condition_instance] = max(condition_maxes[condition_instance], function_value)
        else:
            condition_maxes[condition_instance] = function_value
    d = {}
    for var_values___frozen_dict, function_value in mappings.items():
        if function_value >= condition_maxes[condition_instances[var_values___frozen_dict]]:
            d[var_values___frozen_dict] = function_value
    return d


def discrete_finite_marginalization(discrete_finite_pmf, marginalized_vars):
    var_symbols = deepcopy(discrete_finite_pmf.vars)
    mappings = discrete_finite_pmf.parameters['mappings'].copy()
    for marginalized_var in marginalized_vars:
        del var_symbols[marginalized_var]
        d = {}
        for var_values___frozen_dict, function_value in mappings.items():
            marginalized_var_value = var_values___frozen_dict[marginalized_var]
            fdict = FrozenDict(set(var_values___frozen_dict.items()) - {(marginalized_var, marginalized_var_value)})
            if fdict in d:
                d[fdict] += function_value
            else:
                d[fdict] = function_value
        mappings = d
    return discrete_finite_mass_function(var_symbols, dict(mappings=mappings), deepcopy(discrete_finite_pmf.conditions))


def discrete_finite_conditioning(discrete_finite_pmf, conditions={}):
    mappings = discrete_finite_pmf.parameters['mappings'].copy()
    d = {}
    s0 = set(conditions.items())
    for var_values___frozen_dict, function_value in mappings.items():
        s = set(var_values___frozen_dict.items())
        if s >= s0:
            d[FrozenDict(s - s0)] = function_value
    new_conditions = deepcopy(discrete_finite_pmf.conditions)
    new_conditions.update(conditions)
    return discrete_finite_mass_function(deepcopy(discrete_finite_pmf.vars), dict(mappings=d), new_conditions)


def discrete_finite_mass_function(var_symbols, parameters, conditions={}):
    mappings = parameters['mappings']
    condition_instances = {}
    for var_values___frozen_dict in mappings:
        condition_instance = {}
        for var in (set(var_values___frozen_dict) & set(conditions)):
            condition_instance[var] = var_values___frozen_dict[var]
        condition_instance = FrozenDict(condition_instance)
        condition_instances[var_values___frozen_dict] = condition_instance
    return ProbabilityDensityFunction('DiscreteFinite', deepcopy(var_symbols),
                                      dict(mappings=mappings, condition_instances=condition_instances),
                                      discrete_finite_mass, discrete_finite_normalization, discrete_finite_max,
                                      discrete_finite_marginalization, discrete_finite_conditioning,
                                      deepcopy(conditions))


def gaussian_density(vars_row_vectors___dict, parameters___dict):
    var_names = tuple(vars_row_vectors___dict)
    num_vars = len(var_names)
    x = num_vars * [None]
    m = num_vars * [None]
    S = [num_vars * [None] for _ in range(num_vars)]   # careful not to create same mutable object
    d = 0
    for i in range(num_vars):
        x[i] = vars_row_vectors___dict[var_names[i]]
        d += vars_row_vectors___dict[var_names[i]].shape[1]
        m[i] = parameters___dict[('mean', var_names[i])]
        for j in range(i):
            if ('cov', var_names[i], var_names[j]) in parameters___dict:
                S[i][j] = parameters___dict[('cov', var_names[i], var_names[j])]
                S[j][i] = S[i][j].T
            else:
                S[j][i] = parameters___dict[('cov', var_names[j], var_names[i])]
                S[i][j] = S[j][i].T
        S[i][i] = parameters___dict[('cov', var_names[i])]
    x = BlockMatrix([x])
    m = BlockMatrix([m])
    S = BlockMatrix(S)
    return (d * log(2 * pi) + log(det(S)) + det((x - m) * S.inverse() * (x - m).T)) / 2


def gaussian_max(gaussian_pdf):
    means = {}
    for var in gaussian_pdf.vars:
        means[var] = gaussian_pdf.parameters[('mean', var)]
    return {FrozenDict(means): gaussian_pdf(means)}


def gaussian_marginalization(gaussian_pdf, marginalized_vars):
    var_symbols = deepcopy(gaussian_pdf.vars)
    parameters = deepcopy(gaussian_pdf.parameters)
    for var in marginalized_vars:
        del var_symbols[var]
        for key in gaussian_pdf.parameters:
            if var in key:
                del parameters[key]
    return gaussian_density_function(var_symbols, parameters, deepcopy(gaussian_pdf.conditions))


def gaussian_conditioning(gaussian_pdf, conditions={}):
    new_conditions = deepcopy(gaussian_pdf.conditions)
    new_conditions.update(conditions)
    point_conditions = {}
    for var, value in conditions.items():
        if value is not None:
            point_conditions[gaussian_pdf.vars[var]] = value
    condition_var_names = list(conditions)
    num_condition_vars = len(condition_var_names)
    scope_var_names = list(gaussian_pdf.scope - set(conditions))
    num_scope_vars = len(scope_var_names)
    x_c = num_condition_vars * [None]
    m_c = num_condition_vars * [None]
    m_s = num_scope_vars * [None]
    S_c = [num_condition_vars * [None] for _ in range(num_condition_vars)]   # careful not to create same mutable object
    S_s = [num_scope_vars * [None] for _ in range(num_scope_vars)]   # careful not to create same mutable object
    S_cs = [num_scope_vars * [None] for _ in range(num_condition_vars)]   # careful not to create same mutable object
    for i in range(num_condition_vars):
        x_c[i] = gaussian_pdf.vars[condition_var_names[i]]
        m_c[i] = gaussian_pdf.parameters[('mean', condition_var_names[i])]
        for j in range(i):
            if ('cov', condition_var_names[i], condition_var_names[j]) in gaussian_pdf.parameters:
                S_c[i][j] = gaussian_pdf.parameters[('cov', condition_var_names[i], condition_var_names[j])]
                S_c[j][i] = S_c[i][j].T
            else:
                S_c[j][i] = gaussian_pdf.parameters[('cov', condition_var_names[j], condition_var_names[i])]
                S_c[i][j] = S_c[j][i].T
        S_c[i][i] = gaussian_pdf.parameters[('cov', condition_var_names[i])]
    for i in range(num_scope_vars):
        m_s[i] = gaussian_pdf.parameters[('mean', scope_var_names[i])]
        for j in range(i):
            if ('cov', scope_var_names[i], scope_var_names[j]) in gaussian_pdf.parameters:
                S_s[i][j] = gaussian_pdf.parameters[('cov', scope_var_names[i], scope_var_names[j])]
                S_s[j][i] = S_s[i][j].T
            else:
                S_s[j][i] = gaussian_pdf.parameters[('cov', scope_var_names[j], scope_var_names[i])]
                S_s[i][j] = S_s[j][i].T
        S_s[i][i] = gaussian_pdf.parameters[('cov', scope_var_names[i])]
    for i, j in itertools.product(range(num_condition_vars), range(num_scope_vars)):
        if ('cov', condition_var_names[i], scope_var_names[j]) in gaussian_pdf.parameters:
            S_cs[i][j] = gaussian_pdf.parameters[('cov', condition_var_names[i], scope_var_names[j])]
        else:
            S_cs[i][j] = gaussian_pdf.parameters[('cov', scope_var_names[j], condition_var_names[i])].T
    x_c = BlockMatrix([x_c])
    m_c = BlockMatrix([m_c])
    m_s = BlockMatrix([m_s])
    S_c = BlockMatrix(S_c)
    S_s = BlockMatrix(S_s)
    S_cs = BlockMatrix(S_cs)
    S_sc = S_cs.T
    m = (m_s + (x_c - m_c) * S_c.inverse() * S_cs).xreplace(point_conditions)
    S = S_s - S_sc * S_c.inverse() * S_cs
    parameters = {}
    index_ranges_from = []
    index_ranges_to = []
    k = 0
    for i in range(num_scope_vars):
        l = k + gaussian_pdf.vars[scope_var_names[i]].shape[1]
        index_ranges_from += [k]
        index_ranges_to += [l]
        parameters[('means', scope_var_names[i])] = m[0, index_ranges_from[i]:index_ranges_to[i]]
        for j in range(i):
            parameters[('cov', scope_var_names[j], scope_var_names[i])] =\
                S[index_ranges_from[j]:index_ranges_to[j], index_ranges_from[i]:index_ranges_to[i]]
        parameters[('cov', scope_var_names[i])] =\
            S[index_ranges_from[i]:index_ranges_to[i], index_ranges_from[i]:index_ranges_to[i]]
        k = l
    return gaussian_density_function(deepcopy(gaussian_pdf.vars), parameters, new_conditions)


def gaussian_density_function(var_symbols, parameters, conditions={}):
    return ProbabilityDensityFunction('Gaussian', deepcopy(var_symbols), deepcopy(parameters),
                                      gaussian_density, lambda: None, gaussian_max, gaussian_marginalization,
                                      gaussian_conditioning, deepcopy(conditions))


def product_of_2_discrete_finite_probability_mass_functions(pmf_1, pmf_2):
    conditions = deepcopy(pmf_1.conditions)   # just to be careful #
    scope = deepcopy(pmf_1.scope)   # just to be careful
    conditions.update(pmf_2.conditions)
    scope.update(pmf_2.scope)
    for var in (set(conditions) & scope):
        conditions.pop(var, None)
        scope.add(var)
    var_symbols = deepcopy(pmf_1.vars)
    var_symbols.update(pmf_2.vars)
    mappings_1 = pmf_1.parameters['mappings'].copy()
    mappings_2 = pmf_2.parameters['mappings'].copy()
    mappings = {}
    for item_1, item_2 in itertools.product(mappings_1.items(), mappings_2.items()):
        var_values_1___frozen_dict, function_value_1 = item_1
        var_values_2___frozen_dict, function_value_2 = item_2
        same_vars_same_values = True
        for var in (set(var_values_1___frozen_dict) & set(var_values_2___frozen_dict)):
            same_vars_same_values &= (var_values_1___frozen_dict[var] == var_values_2___frozen_dict[var])
        if same_vars_same_values:
            mappings[FrozenDict(set(var_values_1___frozen_dict.items()) | set(var_values_2___frozen_dict.items()))] =\
                function_value_1 + function_value_2
    return discrete_finite_mass_function(var_symbols, dict(mappings=mappings), conditions)


def product_of_discrete_finite_probability_mass_function_and_continuous_probability_density_function(pmf, pdf):
    conditions = deepcopy(pmf.conditions)   # just to be careful #
    scope = deepcopy(pmf.scope)   # just to be careful
    conditions.update(pdf.conditions)
    scope.update(pdf.scope)
    for var in (set(conditions) & scope):
        conditions.pop(var, None)
        scope.add(var)
    var_symbols = deepcopy(pmf.vars)
    var_symbols.update(pdf.vars)
    mappings = {}
    for var_values___frozen_dict, function_value in pmf.parameters['mappings'].items():
        mappings[var_values___frozen_dict] = function_value + pdf.density_lambda(pdf.vars)
    return discrete_finite_mass_function(var_symbols, dict(mappings=mappings), conditions)


def product_of_2_gaussian_probability_density_functions(gaussian_pdf_1, gaussian_pdf_2):
    conditions = deepcopy(gaussian_pdf_1.conditions)   # just to be careful #
    scope = deepcopy(gaussian_pdf_1.scope)   # just to be careful
    conditions.update(gaussian_pdf_2.conditions)
    scope.update(gaussian_pdf_2.scope)
    for var in (set(conditions) & scope):
        conditions.pop(var, None)
        scope.add(var)
    var_symbols = deepcopy(gaussian_pdf_1.vars)
    var_symbols.update(gaussian_pdf_2.vars)
    parameters = {}
    return gaussian_density_function(var_symbols, parameters, conditions)