from __future__ import print_function
from copy import deepcopy
from frozendict import frozendict
from MathDict import MathDict
from pprint import pprint
from sympy import exp, log, sympify
from Helpy.Dicts import combine_dict_and_kwargs, merge_dicts
from Helpy.SymPy import sympy_dicts_allclose, sympy_prob_from_neg_log_prob,\
    is_non_atomic_sympy_expr, sympy_xreplace, sympy_xreplace_doit_explicit, sympy_xreplace_doit_explicit_eval,\
    shift_time_subscripts


class PDF:
    def __init__(self, family_name, var_names_and_symbols___dict, params___dict, neg_log_density_func,
                 normalization_func, max_func, marginalization_func, conditioning_func, sampling_func,
                 conditions={}, scope={}):
        self.Family = family_name
        self.Vars = var_names_and_symbols___dict
        self.Conditions = conditions
        self.Scope = dict.fromkeys(set(var_names_and_symbols___dict) - set(conditions))
        for var, value in scope.items():
            if (var in self.Scope) and (value is not None):
                self.Scope[var] = value
        self.Params = params___dict
        self.NegLogDensityFunc = neg_log_density_func
        self.NormalizationFunc = normalization_func
        self.MaxFunc = max_func
        self.MarginalizationFunc = marginalization_func
        self.ConditioningFunc = conditioning_func
        self.SamplingFunc = sampling_func

    def copy(self, deep=True):
        if deep:
            return PDF(self.Family, deepcopy(self.Vars), deepcopy(self.Params),
                       self.NegLogDensityFunc, self.NormalizationFunc, self.MaxFunc,
                       self.MarginalizationFunc, self.ConditioningFunc, self.SamplingFunc,
                       deepcopy(self.Conditions), deepcopy(self.Scope))
        else:
            return PDF(self.Family, self.Vars.copy(), self.Params.copy(),
                       self.NegLogDensityFunc, self.NormalizationFunc, self.MaxFunc,
                       self.MarginalizationFunc, self.ConditioningFunc, self.SamplingFunc,
                       self.Conditions.copy(), self.Scope.copy())

    def is_one(self):
        return self.Family == 'One'

    def is_discrete_finite(self):
        return self.Family == 'DiscreteFinite'

    def is_uniform(self):
        return self.Family == 'Uniform'

    def is_gaussian(self):
        return self.Family == 'Gaussian'

    def at(self, var_and_param_values___dict={}, **kw_var_and_param_values___dict):
        var_and_param_values___dict = combine_dict_and_kwargs(var_and_param_values___dict,
                                                              kw_var_and_param_values___dict)
        for var in (set(self.Vars) & set(var_and_param_values___dict)):
            var_and_param_values___dict[self.Vars[var]] = var_and_param_values___dict[var]
        pdf = self.copy()
        for var, value in var_and_param_values___dict.items():
            if var in pdf.Conditions:
                pdf.Conditions.update({var: value})
            if var in pdf.Scope:
                pdf.Scope.update({var: value})
        pdf.Conditions = sympy_xreplace(pdf.Conditions, var_and_param_values___dict)
        pdf.Scope = sympy_xreplace(pdf.Scope, var_and_param_values___dict)
        if pdf.is_discrete_finite():
            neg_log_prob = {}
            for vars_and_values___frozen_dict, neg_log_prob in pdf.Params['NegLogProb'].items():
                other_items___dict = dict(set(vars_and_values___frozen_dict.items()) -
                                          set(var_and_param_values___dict.items()))
                if not (set(other_items___dict) and set(var_and_param_values___dict)):
                    neg_log_prob[frozendict(set(vars_and_values___frozen_dict.items()) -
                                             set(pdf.Conditions.items()))] =\
                        sympy_xreplace(neg_log_prob, var_and_param_values___dict)
            return DiscreteFinitePMF(pdf.Vars, neg_log_prob, conditions=pdf.Conditions, scope=pdf.Scope, prob=False)
        else:
            pdf.Params = sympy_xreplace(pdf.Params, var_and_param_values___dict)
            return pdf

    def __call__(self, var_and_param_values___dict={}, prob=True):
        scope_vars = deepcopy(self.Vars)
        for var in self.Vars:
            if var not in self.Scope:
                del scope_vars[var]
            elif self.Scope[var] is not None:
                scope_vars[var] = self.Scope[var]
        if var_and_param_values___dict:
            symbols_and_values___dict = {}
            for var_or_param, value in var_and_param_values___dict.items():
                if var_or_param in self.Vars:
                    symbols_and_values___dict[self.Vars[var_or_param]] = value
                elif var_or_param in self.Params:
                    symbols_and_values___dict[self.Params[var_or_param]] = value
            if prob:
                return sympy_xreplace_doit_explicit_eval(
                    sympy_prob_from_neg_log_prob(self.NegLogDensityFunc(scope_vars, self.Params)),
                    symbols_and_values___dict)
            else:
                return sympy_xreplace_doit_explicit_eval(self.NegLogDensityFunc(scope_vars, self.Params),
                                                         symbols_and_values___dict)
        elif prob:
            return sympy_prob_from_neg_log_prob(self.NegLogDensityFunc(scope_vars, self.Params))
        else:
            return self.NegLogDensityFunc(scope_vars, self.Params)

    def normalize(self):
        return self.NormalizationFunc(self)

    def max(self, **kwargs):
        return self.MaxFunc(self, **kwargs)

    def marginalize(self, *marginalized_vars):
        return self.MarginalizationFunc(self, *marginalized_vars)

    def condition(self, conditions={}, **kw_conditions):
        conditions = combine_dict_and_kwargs(conditions, kw_conditions)
        return self.ConditioningFunc(self, conditions)

    def sample(self, num_samples=1):
        return self.SamplingFunc(self, num_samples)

    def __mul__(self, pdf_or_other_obj):
        if isinstance(pdf_or_other_obj, PDF):
            return product_of_2_PDFs(self, pdf_or_other_obj)
        else:
            return self * pdf_or_other_obj

    def __rmul__(self, pdf):
        return product_of_2_PDFs(pdf, self)

    def multiply(self, *PDFs):
        pdf = self.copy()
        for pdf_to_multiply in PDFs:
            pdf *= pdf_to_multiply
        return pdf

    def pprint(self):
        discrete_finite = self.is_discrete_finite()
        print('\n')
        if discrete_finite:
            print('MASS FUNCTION')
            print('_____________')
        else:
            print('DENSITY FUNCTION')
            print('________________')
            print('FAMILY:', self.Family)
        print("VARIABLES' SYMBOLS:")
        pprint(self.Vars)
        print('CONDITIONS:')
        pprint(self.Conditions)
        print('SCOPE:')
        pprint(self.Scope)
        if not discrete_finite:
            print('PARAMETERS:')
            pprint(self.Params)
            print('DENSITY:')
        else:
            print('MASS:')
        d = self()
        pprint(d)
        if discrete_finite:
            print('   sum =', sum(d.values()))
        print('\n')

    def shift_time_subscripts(self, t):
        pdf = self.copy()
        pdf.Vars = shift_time_subscripts(pdf.Vars, t)
        pdf.Conditions = shift_time_subscripts(pdf.Conditions, t)
        pdf.Scope = shift_time_subscripts(pdf.Scope, t)
        pdf.Params = shift_time_subscripts(pdf.Params, t)
        return pdf


class DiscreteFinitePMF(PDF):
    def __init__(self, var_names_and_symbols___dict, probs_or_neg_log_probs___dict, conditions={}, scope={}, prob=True):
        if prob:
            neg_log_probs___dict = {k: log(v) for k, v in probs_or_neg_log_probs___dict.items()}
        else:
            neg_log_probs___dict = probs_or_neg_log_probs___dict
        non_none_scope = {var: value for var, value in scope.items() if value is not None}
        mappings = {var_values___frozen_dict: mapping_value
                    for var_values___frozen_dict, mapping_value in neg_log_probs___dict['mappings'].items()
                    if set(var_values___frozen_dict.items()) >= set(non_none_scope.items())}
        condition_instances = {}
        for var_values___frozen_dict in mappings:
            condition_instance = {}
            for var in (set(var_values___frozen_dict) & set(conditions)):
                condition_instance[var] = var_values___frozen_dict[var]
            condition_instances[var_values___frozen_dict] = frozendict(condition_instance)
        PDF.__init__(self, 'DiscreteFinite',
                     var_names_and_symbols___dict.copy(),
                                          dict(NegLogProbs=mappings, ConditionInstances=condition_instances),
                                          discrete_finite_mass, discrete_finite_normalization, discrete_finite_max,
                                          discrete_finite_marginalization, discrete_finite_conditioning,
                                          lambda *args, **kwargs: None, deepcopy(conditions), deepcopy(non_none_scope))


def discrete_finite_mass(var_values___dict, parameters):
    v = deepcopy(var_values___dict)
    for var, value in var_values___dict.items():
        if (value is None) or is_non_atomic_sympy_expr(value):
            del v[var]
    s0 = set(v.items())
    d = {}
    mappings = parameters['mappings']
    for var_values___frozen_dict, mapping_value in mappings.items():
        spare_var_values = dict(s0 - set(var_values___frozen_dict.items()))
        s = set(spare_var_values.keys())
        if not(s) or (s and not(s & set(var_values___frozen_dict))):
            d[var_values___frozen_dict] = sympy_xreplace_doit_explicit(mapping_value, var_values___dict)
    return d


def discrete_finite_normalization(discrete_finite_pmf):
    pmf = discrete_finite_pmf.copy()
    pmf.parameters['mappings'] = pmf.parameters['mappings'].copy()
    mappings = pmf.parameters['mappings']
    condition_instances = pmf.parameters['condition_instances']
    condition_sums = {}
    for var_values___frozen_dict, function_value in mappings.items():
        condition_instance = condition_instances[var_values___frozen_dict]
        if condition_instance in condition_sums:
            condition_sums[condition_instance] += exp(-function_value)
        else:
            condition_sums[condition_instance] = exp(-function_value)
    for var_values___frozen_dict in mappings:
        pmf.parameters['mappings'][var_values___frozen_dict] +=\
            log(condition_sums[condition_instances[var_values___frozen_dict]])
    return pmf


def discrete_finite_max(discrete_finite_pmf, leave_unoptimized=None):
    mappings = discrete_finite_pmf.parameters['mappings']
    if leave_unoptimized:
        comparison_bases = {}
        conditioned_and_unoptimized_vars = set(discrete_finite_pmf.conditions) | set(leave_unoptimized)
        for var_values___frozen_dict in mappings:
            comparison_basis = {}
            for var in (set(var_values___frozen_dict) & conditioned_and_unoptimized_vars):
                comparison_basis[var] = var_values___frozen_dict[var]
            comparison_bases[var_values___frozen_dict] = frozendict(comparison_basis)
    else:
        comparison_bases = discrete_finite_pmf.parameters['condition_instances']
    minus_log_mins = {}
    for var_values___frozen_dict, mapping_value in mappings.items():
        comparison_basis = comparison_bases[var_values___frozen_dict]
        if comparison_basis in minus_log_mins:
            minus_log_mins[comparison_basis] = min(minus_log_mins[comparison_basis], mapping_value)
        else:
            minus_log_mins[comparison_basis] = mapping_value
    max_mappings = {}
    for var_values___frozen_dict, mapping_value in mappings.items():
        if mapping_value <= minus_log_mins[comparison_bases[var_values___frozen_dict]]:
            max_mappings[var_values___frozen_dict] = mapping_value
    return DiscreteFinitePMF(discrete_finite_pmf.vars.copy(), dict(mappings=max_mappings),
                                         deepcopy(discrete_finite_pmf.conditions), deepcopy(discrete_finite_pmf.scope))


def discrete_finite_marginalization(discrete_finite_pmf, *marginalized_vars):
    var_symbols = discrete_finite_pmf.vars.copy()
    mappings = discrete_finite_pmf.parameters['mappings'].copy()
    for marginalized_var in marginalized_vars:
        del var_symbols[marginalized_var]
        d = {}
        for var_values___frozen_dict, mapping_value in mappings.items():
            marginalized_var_value = var_values___frozen_dict[marginalized_var]
            fd = frozendict(set(var_values___frozen_dict.items()) - {(marginalized_var, marginalized_var_value)})
            if fd in d:
                d[fd] += exp(-mapping_value)
            else:
                d[fd] = exp(-mapping_value)
        mappings = {k: -log(v) for k, v in d.items()}
    return DiscreteFinitePMF(var_symbols, dict(mappings=mappings),
                                         deepcopy(discrete_finite_pmf.conditions), deepcopy(discrete_finite_pmf.scope))


def discrete_finite_conditioning(discrete_finite_pmf, conditions={}, **kw_conditions):
    conditions = combine_dict_and_kwargs(conditions, kw_conditions)
    mappings = discrete_finite_pmf.parameters['mappings'].copy()
    d = {}
    s0 = set(conditions.items())
    for var_values___frozen_dict, mapping_value in mappings.items():
        s = set(var_values___frozen_dict.items())
        if s >= s0:
            d[frozendict(s - s0)] = mapping_value
    new_conditions = deepcopy(discrete_finite_pmf.conditions)
    new_conditions.update(conditions)
    scope = deepcopy(discrete_finite_pmf.scope)
    for var in conditions:
        del scope[var]
    return DiscreteFinitePMF(discrete_finite_pmf.vars.copy(), dict(mappings=d), new_conditions, scope)


def discrete_finite_mass_functions_all_close(*pmfs, **kwargs):
    if len(pmfs) == 2:
        pmf_0, pmf_1 = pmfs
        return (set(pmf_0.vars.items()) == set(pmf_1.vars.items())) &\
            (set(pmf_0.conditions.items()) == set(pmf_1.conditions.items())) &\
            (set(pmf_0.scope.items()) == set(pmf_1.scope.items())) &\
            dicts_all_close(pmf_0.parameters['mappings'], pmf_1.parameters['mappings'], **kwargs)
    else:
        for i in range(1, len(pmfs)):
            if not discrete_finite_mass_functions_all_close(pmfs[0], pmfs[i], **kwargs):
                return False
        return True


def one_density_function(var_symbols={}, conditions={}):
    return PDF('One', var_symbols.copy(), {}, one, one, one, one, one,
                                      lambda *args, **kwargs: None, deepcopy(conditions))


def one(*args, **kwargs):
    return sympify(0.)


def one_mass_function(var_symbols, frozen_dicts___set=set(), conditions={}):
    mappings = {item: sympify(0.) for item in frozen_dicts___set}
    return DiscreteFinitePMF(var_symbols, dict(mappings=mappings), conditions, scope={})


def product_of_2_PDFs(pdf_1, pdf_2):
    families = (pdf_1.family, pdf_2.family)
    if families == ('DiscreteFinite', 'DiscreteFinite'):
        return multiply_2_DiscreteFinitePMFs(pdf_1, pdf_2)
    elif pdf_1.is_discrete_finite():
        return multiply_DiscreteFinitePMF_and_continuousPDF(
            pdf_1, pdf_2)
    elif pdf_2.is_discrete_finite():
        return multiply_DiscreteFinitePMF_and_continuousPDF(
            pdf_2, pdf_1)
    elif families == ('One', 'Gaussian'):
        return multiply_OnePDF_and_GaussianPDF(
            pdf_1, pdf_2)
    elif families == ('Gaussian', 'One'):
        return multiply_OnePDF_and_GaussianPDF(
            pdf_2, pdf_1)
    elif families == ('Gaussian', 'Gaussian'):
        return multiply_2_GaussianPDFs(pdf_1, pdf_2)


def multiply_2_DiscreteFinitePMFs(pmf_1, pmf_2):
    conditions = merge_dicts(pmf_1.conditions, pmf_2.conditions)
    scope = merge_dicts(pmf_1.scope, pmf_2.scope)
    for var in (set(conditions) & set(scope)):
        del conditions[var]
    var_symbols = merge_dicts(pmf_1.vars, pmf_2.vars)
    mappings_1 = pmf_1.parameters['mappings'].copy()
    mappings_2 = pmf_2.parameters['mappings'].copy()
    mappings = {}
    for item_1, item_2 in itertools.product(mappings_1.items(), mappings_2.items()):
        var_values_1___frozen_dict, mapping_value_1 = item_1
        var_values_2___frozen_dict, mapping_value_2 = item_2
        same_vars_same_values = True
        for var in (set(var_values_1___frozen_dict) & set(var_values_2___frozen_dict)):
            same_vars_same_values &= (var_values_1___frozen_dict[var] == var_values_2___frozen_dict[var])
        if same_vars_same_values:
            mappings[fdict(set(var_values_1___frozen_dict.items()) | set(var_values_2___frozen_dict.items()))] =\
                mapping_value_1 + mapping_value_2
    return DiscreteFinitePMF(var_symbols, dict(mappings=mappings), conditions, scope)


def multiply_DiscreteFinitePMF_and_continuousPDF(pmf, pdf):
    conditions = merge_dicts(pmf.conditions, pdf.conditions)
    scope = merge_dicts(pmf.scope, pdf.scope)
    for var in (set(conditions) & set(scope)):
        del conditions[var]
    var_symbols = merge_dicts(pmf.vars, pdf.vars)
    mappings = {}
    for var_values___frozen_dict, mapping_value in pmf.parameters['mappings'].items():
        mappings[var_values___frozen_dict] = mapping_value + pdf.density_lambda(pdf.vars)
    return DiscreteFinitePMF(var_symbols, dict(mappings=mappings), conditions, scope)


def multiply_OnePDF_and_GaussianPDF(one_pdf, gaussian_pdf):
    conditions = merge_dicts(gaussian_pdf.conditions, one_pdf.conditions)
    scope = deepcopy(gaussian_pdf.scope, one_pdf.scope)
    for var in (set(conditions) & set(scope)):
        del conditions[var]
    var_symbols = merge_dicts(gaussian_pdf.vars, one_pdf.vars)
    return gaussian_density_function(var_symbols, deepcopy(gaussian_pdf.parameters), conditions, scope)


def multiply_2_GaussianPDFs(gaussian_pdf_1, gaussian_pdf_2):
    conditions = merge_dicts(gaussian_pdf_1.conditions, gaussian_pdf_2.conditions)
    scope = merge_dicts(gaussian_pdf_1.scope, gaussian_pdf_2.scope)
    for var in (set(conditions) & set(scope)):
        del conditions[var]
    var_symbols = merge_dicts(gaussian_pdf_1.vars, gaussian_pdf_2.vars)
    parameters = {}
    return gaussian_density_function(var_symbols, parameters, conditions, scope)


from __future__ import division
from copy import deepcopy
from itertools import product
from scipy.stats import multivariate_normal
from sympy import log, pi
from sympy.matrices import BlockMatrix, det
from _BaseClasses import PDF, one_density_function
from zzzUtils import combine_dict_and_kwargs


class GaussianPDF:
    def __init__(var_symbols, params___dict, conditions={}, scope={}):
        return PDF('Gaussian', var_symbols.copy(), deepcopy(params___dict),
                                          gaussian_density, lambda *args, **kwargs: None, gaussian_max,
                                          gaussian_marginalize, gaussian_condition, gaussian_sample,
                                          deepcopy(conditions), deepcopy(scope))


def gaussian_density(var_row_vectors___dict, params___dict):
    var_names = tuple(var_row_vectors___dict)
    num_vars = len(var_names)
    x = []
    m = []
    S = [num_vars * [None] for _ in range(num_vars)]   # careful not to create same mutable object
    d = 0
    for i in range(num_vars):
        x += [var_row_vectors___dict[var_names[i]]]
        d += var_row_vectors___dict[var_names[i]].shape[1]
        m += [params___dict[('mean', var_names[i])]]
        for j in range(i):
            if (var_names[i], var_names[j]) in params___dict.cov:
                S[i][j] = params___dict[('cov', var_names[i], var_names[j])]
                S[j][i] = S[i][j].T
            else:
                S[j][i] = params___dict[('cov', var_names[j], var_names[i])]
                S[i][j] = S[j][i].T
        S[i][i] = params___dict.cov[var_names[i]]
    x = BlockMatrix([x])
    m = BlockMatrix([m])
    S = BlockMatrix(S)
    return (d * log(2 * pi) + log(det(S)) + det((x - m) * S.inverse() * (x - m).T)) / 2


def gaussian_max(gaussian_pdf):
    pdf = gaussian_pdf.copy()
    for var, value in gaussian_pdf.scope.items():
        if value is None:
            pdf.scope[var] = pdf.parameters.mean[var]
    return pdf


def gaussian_marginalize(gaussian_pdf, *marginalized_vars):
    vars_and_symbols = gaussian_pdf.vars.copy()
    var_scope = deepcopy(gaussian_pdf.scope)
    parameters = deepcopy(gaussian_pdf.parameters)
    for marginalized_var in marginalized_vars:
        del vars_and_symbols[marginalized_var]
        del var_scope[marginalized_var]
        p = deepcopy(parameters)
        for key in p:
            if marginalized_var in key:
                del parameters[key]
    if var_scope:
        return GaussianPDF(vars_and_symbols, parameters, deepcopy(gaussian_pdf.conditions), var_scope)
    else:
        return one_density_function(vars_and_symbols, deepcopy(gaussian_pdf.conditions))


def gaussian_condition(gaussian_pdf, conditions={}, **kw_conditions):
    conditions = combine_dict_and_kwargs(conditions, kw_conditions)
    new_conditions = deepcopy(gaussian_pdf.conditions)
    new_conditions.update(conditions)
    scope = deepcopy(gaussian_pdf.scope)
    for var in conditions:
        del scope[var]
    point_conditions = {}
    for var, value in conditions.items():
        if value is not None:
            point_conditions[gaussian_pdf.vars[var]] = value
    condition_var_names = list(conditions)
    num_condition_vars = len(condition_var_names)
    scope_var_names = list(set(gaussian_pdf.scope) - set(conditions))
    num_scope_vars = len(scope_var_names)
    x_c = []
    m_c = []
    m_s = []
    S_c = [num_condition_vars * [None] for _ in range(num_condition_vars)]   # careful not to create same mutable object
    S_s = [num_scope_vars * [None] for _ in range(num_scope_vars)]   # careful not to create same mutable object
    S_cs = [num_scope_vars * [None] for _ in range(num_condition_vars)]   # careful not to create same mutable object
    for i in range(num_condition_vars):
        x_c += [gaussian_pdf.vars[condition_var_names[i]]]
        m_c += [gaussian_pdf.parameters[('mean', condition_var_names[i])]]
        for j in range(i):
            if ('cov', condition_var_names[i], condition_var_names[j]) in gaussian_pdf.parameters:
                S_c[i][j] = gaussian_pdf.parameters[('cov', condition_var_names[i], condition_var_names[j])]
                S_c[j][i] = S_c[i][j].T
            else:
                S_c[j][i] = gaussian_pdf.parameters[('cov', condition_var_names[j], condition_var_names[i])]
                S_c[i][j] = S_c[j][i].T
        S_c[i][i] = gaussian_pdf.parameters[('cov', condition_var_names[i])]
    for i in range(num_scope_vars):
        m_s += [gaussian_pdf.parameters[('mean', scope_var_names[i])]]
        for j in range(i):
            if ('cov', scope_var_names[i], scope_var_names[j]) in gaussian_pdf.parameters:
                S_s[i][j] = gaussian_pdf.parameters[('cov', scope_var_names[i], scope_var_names[j])]
                S_s[j][i] = S_s[i][j].T
            else:
                S_s[j][i] = gaussian_pdf.parameters[('cov', scope_var_names[j], scope_var_names[i])]
                S_s[i][j] = S_s[j][i].T
        S_s[i][i] = gaussian_pdf.parameters[('cov', scope_var_names[i])]
    for i, j in product(range(num_condition_vars), range(num_scope_vars)):
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
        parameters[('mean', scope_var_names[i])] = m[0, index_ranges_from[i]:index_ranges_to[i]]
        for j in range(i):
            parameters[('cov', scope_var_names[j], scope_var_names[i])] =\
                S[index_ranges_from[j]:index_ranges_to[j], index_ranges_from[i]:index_ranges_to[i]]
        parameters[('cov', scope_var_names[i])] =\
            S[index_ranges_from[i]:index_ranges_to[i], index_ranges_from[i]:index_ranges_to[i]]
        k = l
    return GaussianPDF(deepcopy(gaussian_pdf.vars), parameters,
                                     new_conditions, scope)


def gaussian_sample(gaussian_pdf, num_samples):
#    scope_vars
#    for scope
#
#    scope_vars = tuple(gaussian_pdf.scope)
#
#    num_scope_vars = len(scope_vars)
#    m = []
#    S = [num_scope_vars * [None] for _ in range(num_scope_vars)]   # careful not to create same mutable object
#    for i in range(num_scope_vars):
#        m += [gaussian_pdf.parameters[('mean', scope_vars[i])]]
#        for j in range(i):
#            if ('cov', scope_vars[i], scope_vars[j]) in gaussian_pdf.parameters:
#                S[i][j] = gaussian_pdf.parameters[('cov', scope_vars[i], scope_vars[j])]
#                S[j][i] = S[i][j].T
#            else:
#                S[j][i] = gaussian_pdf.parameters[('cov', scope_vars[j], scope_vars[i])]
#                S[i][j] = S[j][i].T
#        S[i][i] = gaussian_pdf.parameters[('cov', scope_vars[i])]
#    m = BlockMatrix([m]).as_explicit().tolist()[0]
#    S = BlockMatrix(S).as_explicit().tolist()
#    X = multivariate_normal(m, S)
#    samples = X.rvs(num_samples)
#    densities = X.pdf(samples)
#    mappings = {}
#    for i in range(num_samples):
#        fd = {}
#        k = 0
#        for j in range(num_scope_vars):
#            scope_var = scope_vars[j]
#            l = k + gaussian_pdf.vars[scope_var].shape[1]
#            fd[scope_var] = samples[i, k:l]
#        mappings[FrozenDict(fd)] = densities[i]
    return 0 #discrete_finite_mass_function(deepcopy(gaussian_pdf.vars), dict(mappings=mappings),
#                                         deepcopy(gaussian_pdf.conditions))


from scipy.stats import uniform


def uniform_density_function(var_symbols, parameters, conditions={}, scope={}):
    return PDF('Uniform', deepcopy(var_symbols), deepcopy(parameters),
                                      uniform_density, uniform_normalization, lambda *args, **kwargs: None,
                                      uniform_marginalization, uniform_conditioning, uniform_sampling,
                                      deepcopy(conditions), deepcopy(scope))


def uniform_density(var_symbols, parameters):
    d = 1.
    return d


def uniform_normalization():
    return 0


def uniform_marginalization():
    return 0


def uniform_conditioning():
    return 0


def uniform_sampling():
    return 0