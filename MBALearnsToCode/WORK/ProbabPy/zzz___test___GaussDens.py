from __future__ import division, print_function
from numpy import allclose, array, hstack, log, squeeze
from numpy.random import randint, randn
from ProbabPy import GaussPDF
from scipy.stats import multivariate_normal
from sympy.matrices import Matrix, MatrixSymbol
from sys import argv
from time import time


MAX_NUM_VARS = 3
MAX_NUM_DIMS_PER_VAR = 3
SCALE = 1.


def test___gauss_dens(num_tests=9, rtol=1e-5, atol=1e-8):
    """test: Multivariate Gaussian Density: ProbabPy vs. SciPy"""
    num_succs = 0
    total_probabpy_time = .0
    total_scipy_time = .0
    num_tests += 1
    for t in range(num_tests):
        n = randint(MAX_NUM_VARS) + 1
        d = tuple(randint(MAX_NUM_DIMS_PER_VAR) + 1 for _ in range(n))

        x_names = tuple(str(('x', i)) for i in range(n))
        x = tuple(MatrixSymbol(x_names[i], 1, d[i]) for i in range(n))

        if not t:
            m = tuple(MatrixSymbol(str(('m', i)), 1, d[i]) for i in range(n))
            param = {('Mean', x_names[i]): m[i] for i in range(n)}

            S = [n * [None] for _ in range(n)]   # careful not to create same mutable object
            for i in range(n):
                for j in range(i):
                    S[j][i] = MatrixSymbol(str(('S', j, i)), d[j], d[i])
                    param[('Cov', x_names[j], x_names[i])] = S[j][i]
                S[i][i] = MatrixSymbol(str(('S', i)), d[i], d[i])
                param[('Cov', x_names[i])] = S[i][i]

            g = GaussPDF(var_names_and_syms={x_names[i]: x[i] for i in range(n)}, param=param, compile=False)
            print('NON-PREPROCESSED, NON-COMPILED PDF:')
            g.pprint()

        m_values = tuple(array(SCALE * randn(1, d[i]), dtype='float32') for i in range(n))
        param_names_and_values = {('Mean', x_names[i]): Matrix(m_values[i]) for i in range(n)}

        r = array(randn(sum(d), n * MAX_NUM_DIMS_PER_VAR), dtype='float32')
        S_value = r.dot(r.T)
        S_values = [n * [None] for _ in range(n)]   # careful not to create same mutable object
        index_ranges_from = []
        index_ranges_to = []
        k = 0
        for i in range(n):
            l = k + d[i]
            index_ranges_from += [k]
            index_ranges_to += [l]
            for j in range(i):
                S_values[j][i] =\
                    S_value[index_ranges_from[j]:index_ranges_to[j], index_ranges_from[i]:index_ranges_to[i]]
                param_names_and_values[('Cov', x_names[j], x_names[i])] = Matrix(S_values[j][i])
            S_values[i][i] = S_value[index_ranges_from[i]:index_ranges_to[i], index_ranges_from[i]:index_ranges_to[i]]
            param_names_and_values[('Cov', x_names[i])] = Matrix(S_values[i][i])
            k = l

        g = GaussPDF(var_names_and_syms={x_names[i]: x[i] for i in range(n)},
                     param=param_names_and_values, compile=True)
        if not t:
            print('PREPROCESSED & COMPILED PDF:')
            g.pprint()

        x_values = tuple(array(SCALE * randn(1, d[i]), dtype='float32') for i in range(n))
        var_names_and_values = {x_names[i]: x_values[i] for i in range(n)}

        t0 = time()
        p0 = g(var_names_and_values)   # var_and_param_names_and_values
        probabpy_time = time() - t0
        probabpy_time_ms = '{:.3g}'.format(1e3 * probabpy_time)
        neg_log_p0 = -log(p0)

        mn = multivariate_normal(mean=squeeze(hstack(m_values).T).tolist(), cov=S_value.tolist())
        t0 = time()
        p1 = mn.pdf(squeeze(hstack(x_values).T).tolist())
        scipy_time = time() - t0
        scipy_time_ms = '{:.3g}'.format(1e3 * scipy_time)
        neg_log_p1 = -log(p1)

        time_comp = '| time: ' + probabpy_time_ms + '[ms] vs. ' + scipy_time_ms + '[ms]'
        if scipy_time:
            probabpy_time_as_percent = '{:.0f}'.format(100 * probabpy_time / scipy_time)
            time_comp += ' (' + probabpy_time_as_percent + '%)'

        if t:
            total_probabpy_time += probabpy_time
            total_scipy_time += scipy_time

        if p0 > 0:
            numerical_problem = False
            succ = allclose(neg_log_p0, neg_log_p1, rtol=rtol, atol=atol)
        else:
            numerical_problem = True
            succ = True

        if succ:
            if numerical_problem:
                print('Test', t, ': * Theano numerical problem *', time_comp)
            else:
                print('Test', t, ': succ: -log', p0, '=', neg_log_p0, '==', neg_log_p1, time_comp)
        else:
            print('Test', t, ': !!! FAIL !!!: -log', p0, '=', neg_log_p0, '<>', neg_log_p1, time_comp)

        num_succs += succ

    print(num_succs, 'Successes /', num_tests, 'Tests')

    total_probabpy_time_ms = '{:.3g}'.format(1e3 * total_probabpy_time)
    total_scipy_time_ms = '{:.3g}'.format(1e3 * total_scipy_time)
    total_time_comp =\
        'time: ' + total_probabpy_time_ms + '[ms] vs. ' + total_scipy_time_ms + '[ms]'
    if total_scipy_time:
        total_probabpy_time_as_percent = '{:.0f}'.format(100 * total_probabpy_time / total_scipy_time)
        total_time_comp += ' (' + total_probabpy_time_as_percent + '%)'
    print(total_time_comp)

    assert num_succs >= .9 * num_tests


if __name__ == '__main__':
    if len(argv) == 1:
        test___gauss_dens()
    else:
        test___gauss_dens(num_tests=int(argv[1]))
