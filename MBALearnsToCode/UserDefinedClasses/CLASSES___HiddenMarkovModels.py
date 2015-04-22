import sympy
from frozen_dict import FrozenDict
from MBALearnsToCode.UserDefinedClasses.CLASSES___DiscreteFunctions import DiscreteFiniteDomainFunction as DFDF,\
    is_discrete_finite_domain_function
from MBALearnsToCode.UserDefinedClasses.CLASSES___ProbabilisticFactors import Factor


class HMM(object):
    def __init__(self, state_var_name_and_next_state_var_name, observation_var_name, state_domain,
                 state_prior, state_transition_likelihood, observation_likelihood,
                 sum_or_integrate='sum'):
        self.state_var_name, self.next_state_var_name = state_var_name_and_next_state_var_name
        self.observation_var_name = observation_var_name
        self.state_domain = state_domain
        self.state_prior = state_prior
        self.state_transition_likelihood = state_transition_likelihood
        self.observation_likelihood = observation_likelihood
        self.sum_or_integrate = sum_or_integrate

    def state_prior_factor(self):
        d = {}
        for state_var_name_and_value, factor_value in self.state_prior.function.discrete_finite_mappings.items():
            state_var_value = state_var_name_and_value[self.state_var_name]
            d[FrozenDict({(self.state_var_name, 0): state_var_value})] = factor_value
        function = DFDF(d)
        return Factor(function)

    def state_transition_factor(self, t):
        d = {}
        for state_var_names_and_values____frozen_dict, factor_value in\
                self.state_transition_likelihood.function.discrete_finite_mappings.items():
            state_var_value = state_var_names_and_values____frozen_dict[self.state_var_name]
            next_state_var_value = state_var_names_and_values____frozen_dict[self.next_state_var_name]
            d[FrozenDict({(self.state_var_name, t - 1): state_var_value,
                          (self.state_var_name, t): next_state_var_value})] = factor_value
        function = DFDF(d)
        return Factor(function, conditions={(self.state_var_name, t - 1): None})

    def observation_factor(self, t):
        d = {}
        for state_var_name_and_value_and_observation_var_name_and_value, factor_value in\
                self.observation_likelihood.function.discrete_finite_mappings.items():
            state_var_value = state_var_name_and_value_and_observation_var_name_and_value[self.state_var_name]
            observation_var_value =\
                state_var_name_and_value_and_observation_var_name_and_value[self.observation_var_name]
            d[FrozenDict({(self.state_var_name, t): state_var_value,
                          (self.observation_var_name, t): observation_var_value})] = factor_value
        function = DFDF(d)
        return Factor(function, conditions={(self.state_var_name, t): None})

    def forward_factor(self, t___list, observations___dict={}):
        if isinstance(t___list, int):
            t = t___list
            if t == 0:
                f = self.state_prior_factor().multiply(self.observation_factor(t))
                if t in observations___dict:
                    f = f.subs({(self.observation_var_name, t): observations___dict[t]})
                return f
            else:
                f = self.forward_factor(t - 1, observations___dict)\
                    .multiply(self.state_transition_factor(t))\
                    .eliminate((((self.state_var_name, t - 1), self.sum_or_integrate, self.state_domain),))\
                    .multiply(self.observation_factor(t))
                if t in observations___dict:
                    f = f.subs({(self.observation_var_name, t): observations___dict[t]})
                return f
        elif isinstance(t___list, (list, range, tuple)):
            d = {}
            t = 0
            f = [self.state_prior_factor().multiply(self.observation_factor(t))]
            if t in observations___dict:
                f[t] = f[t].subs({(self.observation_var_name, t): observations___dict[t]})
            if t in t___list:
                d[t] = f[t]
            for t in range(1, max(t___list) + 1):
                f += [f[t - 1]
                    .multiply(self.state_transition_factor(t))
                    .eliminate((((self.state_var_name, t - 1), self.sum_or_integrate, self.state_domain),))
                    .multiply(self.observation_factor(t))]
                if t in observations___dict:
                    f[t] = f[t].subs({(self.observation_var_name, t): observations___dict[t]})
                if t in t___list:
                    d[t] = f[t]
            return d

    def backward_factor(self, t___list, observations___dict={}, max_t=0):
        T = max(max(observations___dict.keys()), max_t)
        if isinstance(t___list, int):
            t = t___list
            if t == T:
                d = {}
                for value in self.state_domain:
                    d[FrozenDict({(self.state_var_name, T): value})] = 1
                return Factor(DFDF(d), conditions={(self.state_var_name, T): None})
            else:
                b = self.state_transition_factor(t + 1)\
                    .multiply(self.observation_factor(t + 1))
                if (t + 1) in observations___dict:
                    b = b.subs({(self.observation_var_name, t + 1): observations___dict[t + 1]})
                b = b.multiply(self.backward_factor(t + 1, observations___dict))\
                    .eliminate((((self.state_var_name, t + 1), self.sum_or_integrate, self.state_domain),))
                return b
        elif isinstance(t___list, (list, range, tuple)):
            d = {}
            t = T
            temp_dict = {}
            for value in self.state_domain:
                temp_dict[FrozenDict({(self.state_var_name, t): value})] = 1
            b = {t: Factor(DFDF(temp_dict), conditions={(self.state_var_name, t): None})}
            if t in t___list:
                d[t] = b[t]
            for t in reversed(range(min(t___list), T)):
                b[t] = self.state_transition_factor(t + 1)\
                    .multiply(self.observation_factor(t + 1))
                if (t + 1) in observations___dict:
                    b[t] = b[t].subs({(self.observation_var_name, t + 1): observations___dict[t + 1]})
                b[t] = b[t].multiply(b[t + 1])\
                    .eliminate((((self.state_var_name, t + 1), self.sum_or_integrate, self.state_domain),))
                if t in t___list:
                    d[t] = b[t]
            return d

    def infer_state(self, t___list, observations___dict={}):
        conditions___dict = {}
        for t, value in observations___dict.items():
            conditions___dict[(self.observation_var_name, t)] = value
        if isinstance(t___list, int):
            t = t___list
            return self.forward_factor(t, observations___dict)\
                .multiply(self.backward_factor(t, observations___dict))\
                .condition(None, conditions___dict)\
                .normalize()
        elif isinstance(t___list, (list, range, tuple)):
            d = {}
            forward = self.forward_factor(t___list, observations___dict)
            backward = self.backward_factor(t___list, observations___dict)
            for t in t___list:
                d[t] = forward[t]\
                    .multiply(backward[t])\
                    .condition(None, conditions___dict)\
                    .normalize()
            return d

    def map_joint_distributions(self, observations___list, recursive=False):
        T = len(observations___list) - 1
        if T == 0:
            f = self.state_prior_factor()\
                .multiply(self.observation_factor(0)
                          .subs({(self.observation_var_name, 0): observations___list[0]}))
            if recursive:
                return f
            else:
                return f.max()
        else:
            last_observation = observations___list.pop(T)
            f = (self.state_transition_factor(T)
                 .multiply(self.observation_factor(T)
                           .subs({(self.observation_var_name, T): last_observation}))).max()
            return (self.map_joint_distributions(observations___list, True)
                    .multiply(f)).max()

    def map_state_sequences(self, observations___list):
        T = len(observations___list)
        m = self.map_joint_distributions(observations___list)
        s = set()
        for args_and_values___frozen_dict in m.function.discrete_finite_mappings:
            l = []
            for t in range(T):
                l += [args_and_values___frozen_dict[(self.state_var_name, t)]]
            s.add(tuple(l))
        return s