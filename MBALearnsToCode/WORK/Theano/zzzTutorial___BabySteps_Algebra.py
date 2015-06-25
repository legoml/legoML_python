from __future__ import division, print_function
from numpy import array
from theano import tensor as T, function, pp


# http://deeplearning.net/software/theano/tutorial/adding.html


# Adding two Scalars
x = T.dscalar('x')
print(type(x))
print(x.type)
y = T.dscalar('y')
print(y.type)
z = x + y
print(pp(z))
f = function([x, y], z)
print(f(2, 3))
print(f(16.3, 12.1))
print(z.eval({x: 16.3, y: 12.1}))


# Adding two Matrices
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
print(f([[1, 2], [3, 4]], [[10, 20], [30, 40]]))
print(f(array([[1, 2], [3, 4]]), array([[10, 20], [30, 40]])))


# Exercise
a = T.vector()  # declare variable
out = a + a ** 10               # build symbolic expression
f = function([a], out)   # compile function
print(f([0, 1, 2]))  # prints `array([0, 2, 1026])

a = T.vector()
b = T.vector()  # declare variable
out = a ** 2 + b ** 2 + 2 * a * b  # build symbolic expression
f = function([a, b], out)   # compile function
print(f([1, 2], [4, 5]))  # prints [ 25.  49.]

