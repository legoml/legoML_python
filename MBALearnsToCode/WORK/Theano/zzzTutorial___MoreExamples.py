from __future__ import division, print_function
from numpy import array
from pprint import pprint
from theano import tensor as T, function, Param, shared
from theano.tensor.shared_randomstreams import RandomStreams
#from theano.sandbox.rng_mrg import normal

# http://deeplearning.net/software/theano/tutorial/examples.html


# Logistic Function
x = T.dmatrix('x')
s = 1. / (1. + T.exp(-x))
logistic = function([x], s)
print(logistic([[0, 1], [-1, -2]]))

s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = function([x], s2)
print(logistic2([[0, 1], [-1, -2]]))


# Computing More than one Thing at the Same Time
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff ** 2
f = function([a, b], [diff, abs_diff, diff_squared])
pprint(f([[1, 1], [1, 1]], [[0, 1], [2, 3]]))


# Setting a Default Value for an Argument
x, y = T.dscalars('x', 'y')
z = x + y
f = function([x, Param(y, default=1)], z)
print(f(33))
print(f(33, 2))

x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
f = function([x, Param(y, default=1), Param(w, default=2, name='w_by_name')], z)
print(f(33))
print(f(33, 2))
print(f(33, 0, 1))
print(f(33, w_by_name=1))
print(f(33, w_by_name=1, y=0))


# Using Shared Variables
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates={state: state+inc})
print(state.get_value())
print(accumulator(1))
print(state.get_value())
print(accumulator(300))
print(state.get_value())

state.set_value(-1)
print(accumulator(3))
print(state.get_value())

decrementor = function([inc], state, updates=[(state, state-inc)])
print(decrementor(2))
print(state.get_value())

fn_of_state = state * 2 + inc
# The type of foo must match the shared variable we are replacing
# with the ``givens``
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state,
                       givens=[(state, foo)])
print(skip_shared(1, 3))  # we're using 3 for the state, not state.value
print(state.get_value())  # old state still there, but we didn't use it


# Using Random Numbers

# Brief Example
srng = RandomStreams(seed=234)
rv_u = srng.uniform((2, 2))
rv_n = srng.normal((2, 2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)   # Not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

f_val0 = f()
print(f_val0)
f_val1 = f()  #different numbers from f_val0
print(f_val1)

g_val0 = g()  # different numbers from f_val0 and f_val1
print(g_val0)
g_val1 = g()  # same numbers as g_val0!
print(g_val1)