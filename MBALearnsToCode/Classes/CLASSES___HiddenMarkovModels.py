from copy import deepcopy
from frozen_dict import FrozenDict
from MBALearnsToCode.Classes.CLASSES___ProbabilityDensityFunctions import one_mass_function, one_density_function


class HiddenMarkovModel:
    def __init__(self, state_var, observation_var,
                 state_prior_pdf, transition_pdf_template, observation_pdf_template):
        self.state_var = state_var
        self.observation_var = observation_var
        self.state_prior_pdf = state_prior_pdf
        self.transition_pdf_template = transition_pdf_template
        self.observation_pdf_template = observation_pdf_template

    def transition_pdf(self, t):
        return self.transition_pdf_template.shift_time_subscripts(t)

    def observation_pdf(self, t):
        return self.observation_pdf_template.shift_time_subscripts(t)

    def forward_pdf(self, t___list, observations___dict={}):
        if isinstance(t___list, int):
            t = t___list
            if t == 0:
                f = self.state_prior_pdf.multiply(self.observation_pdf(t))
                if t in observations___dict:
                    f = f.at({(self.observation_var, t): observations___dict[t]})
                return f
            else:
                f = self.forward_pdf(t - 1, observations___dict)\
                    .multiply(self.transition_pdf(t))\
                    .marginalize(((self.state_var, t - 1),))\
                    .multiply(self.observation_pdf(t))
                if t in observations___dict:
                    f = f.at({(self.observation_var, t): observations___dict[t]})
                return f
        elif isinstance(t___list, (list, range, tuple)):
            d = {}
            t = 0
            f = [self.state_prior_pdf.multiply(self.observation_pdf(t))]
            if t in observations___dict:
                f[t] = f[t].at({(self.observation_var, t): observations___dict[t]})
            if t in t___list:
                d[t] = f[t]
            for t in range(1, max(t___list) + 1):
                f += [f[t - 1]
                    .multiply(self.transition_pdf(t))
                    .marginalize(((self.state_var, t - 1),))
                    .multiply(self.observation_pdf(t))]
                if t in observations___dict:
                    f[t] = f[t].at({(self.observation_var, t): observations___dict[t]})
                if t in t___list:
                    d[t] = f[t]
            return d

    def backward_factor(self, t___list, observations___dict={}, max_t=0):
        T = max(max(observations___dict.keys()), max_t)
        if isinstance(t___list, int):
            t = t___list
            if t == T:
                observation_var_symbol = {(self.observation_var, t):
                                          self.observation_pdf(t).vars[(self.observation_var, t)]}
                if self.observation_pdf_template.family == 'DiscreteFinite':
                    var_values___frozen_dicts = self.observation_pdf(t).parameters['mappings'].keys()
                    observation_var_domain =\
                        set(FrozenDict({(self.observation_var, t): var_values___frozen_dict[(self.observation_var, t)]})
                            for var_values___frozen_dict in var_values___frozen_dicts)
                    return one_mass_function(observation_var_symbol, observation_var_domain,
                                             {(self.observation_var, t): None})
                else:
                    return one_density_function(observation_var_symbol, {(self.observation_var, t): None})
            else:
                b = self.transition_pdf(t + 1)\
                    .multiply(self.observation_pdf(t + 1))
                if (t + 1) in observations___dict:
                    b = b.at({(self.observation_var, t + 1): observations___dict[t + 1]})
                b = b.multiply(self.backward_factor(t + 1, observations___dict))\
                    .marginalize(((self.state_var, t + 1),))
                return b
        elif isinstance(t___list, (list, range, tuple)):
            d = {}
            t = T
            observation_var_symbol = {(self.observation_var, t):
                                      self.observation_pdf(t).vars[(self.observation_var, t)]}
            if self.observation_pdf_template.family == 'DiscreteFinite':
                var_values___frozen_dicts = self.observation_pdf(t).parameters['mappings'].keys()
                observation_var_domain =\
                    set(FrozenDict({(self.observation_var, t): var_values___frozen_dict[(self.observation_var, t)]})
                        for var_values___frozen_dict in var_values___frozen_dicts)
                b = {t: one_mass_function(observation_var_symbol, observation_var_domain,
                                             {(self.observation_var, t): None})}
            else:
                b = {t: one_density_function(observation_var_symbol, {(self.observation_var, t): None})}
            if t in t___list:
                d[t] = b[t]
            for t in reversed(range(min(t___list), T)):
                b[t] = self.transition_pdf(t + 1)\
                    .multiply(self.observation_pdf(t + 1))
                if (t + 1) in observations___dict:
                    b[t] = b[t].at({(self.observation_var, t + 1): observations___dict[t + 1]})
                b[t] = b[t].multiply(b[t + 1])\
                    .marginalize(((self.state_var, t + 1),))
                if t in t___list:
                    d[t] = b[t]
            return d

    def infer_state(self, t___list, observations___dict={}):
        conditions___dict = {}
        for t, value in observations___dict.items():
            conditions___dict[(self.observation_var, t)] = value
        if isinstance(t___list, int):
            t = t___list
            return self.forward_pdf(t, observations___dict)\
                .multiply(self.backward_factor(t, observations___dict))\
                .condition(conditions___dict)\
                .normalize()
        elif isinstance(t___list, (list, range, tuple)):
            d = {}
            forward = self.forward_pdf(t___list, observations___dict)
            backward = self.backward_factor(t___list, observations___dict)
            for t in t___list:
                d[t] = forward[t]\
                    .multiply(backward[t])\
                    .condition(conditions___dict)\
                    .normalize()
            return d

    def map_joint_distributions(self, observations___list, recursive=False):
        observations___list = deepcopy(observations___list)
        T = len(observations___list) - 1
        if T == 0:
            f = self.state_prior_pdf\
                .multiply(self.observation_pdf(0)
                          .at({(self.observation_var, 0): observations___list[0]}))
            if recursive:
                return f
            else:
                return f.max()
        else:
            last_observation = observations___list.pop(T)
            f = (self.transition_pdf(T)
                 .multiply(self.observation_pdf(T)
                           .at({(self.observation_var, T): last_observation}))).max()
            return (self.map_joint_distributions(observations___list, True)
                    .multiply(f)).max()

    def map_state_sequences(self, observations___list):
        m = self.map_joint_distributions(observations___list)
        if m.family == 'DiscreteFinite':
            d = set(m.parameters['mappings']).pop()
        else:
            d = m.scope
        return [d[(self.state_var, t)] for t in range(len(observations___list))]