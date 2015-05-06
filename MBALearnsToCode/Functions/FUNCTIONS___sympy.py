def sympy_xreplace_doit_explicit(sympy_expression, xreplace_dict={}):
    sympy_expression = sympy_expression.copy()
    if isinstance(sympy_expression, dict):
        sympy_expression = {k: sympy_xreplace_doit_explicit(v, xreplace_dict) for k, v in sympy_expression.items()}
    else:
        # xreplace into all nodes of the expression tree first
        if xreplace_dict:
            sympy_expression = sympy_expression.xreplace(xreplace_dict)
        # traverse the tree to compute
        if hasattr(sympy_expression, 'args') and sympy_expression.args:
            args = []
            for arg in sympy_expression.args:
                # compute each argument
                args += [sympy_xreplace_doit_explicit(arg)]
            # reconstruct function
            sympy_expression = sympy_expression.func(*args)
            # try to do it if expression is complete
            try:
                sympy_expression = sympy_expression.doit()
            except:
                pass
            # try to make it explicit if possible
            try:
                sympy_expression = sympy_expression.as_explicit()
            except:
                pass
    return sympy_expression


def sympy_xreplace_doit_explicit_evalf(sympy_expression, xreplace_dict={}):
    sympy_expression = sympy_xreplace_doit_explicit(sympy_expression, xreplace_dict)
    # try evaluating out to get numerical value
    if isinstance(sympy_expression, dict):
        for k, v in sympy_expression.items():
            try:
                sympy_expression[k] = v.evalf()
            except:
                pass
    else:
        try:
            sympy_expression = sympy_expression.evalf()
        except:
            pass
    return sympy_expression


def is_sympy_expression(sympy_expression):
    if hasattr(sympy_expression, 'doit'):
        if sympy_expression.args:
            return True
        else:
            return False
    else:
        return False