from numpy import atleast_2d, atleast_3d, diag, exp, squeeze, zeros



def softmax(inputs):
    """softmax:

    Returns the softmax function of INPUTS matrix with cases in rows
    """

    inp = inputs - inputs.max(1, keepdims = True)
    exp_matrix = exp(inp)
    return exp_matrix / exp_matrix.sum(1, keepdims = True)



def softmax_d_outputs_over_d_inputs(inputs = None, outputs = None):
    """softmax_d_outputs_over_d_inputs:

    Returns a 3-dimensional array capturing the partial derivatives of the OUTPUTS matrix (with cases in rows) w.r.t.
    the INPUTS matrix (with cases in rows) of the softmax function
    """

    if outputs is None:
        outputs = softmax(inputs)
    m, n = outputs.shape
    d = zeros([m, n, n])
    for i in range(m):
        outp = outputs[i]
        outp_2d = atleast_2d(outp)
        d[i] = atleast_3d(diag(outp) - (outp_2d.T).dot(outp_2d)).transpose(2, 0, 1)
    return d

def softmax_doverd_inputs_from_doverd_outputs(doverd_outputs, d_outputs_over_d_inputs):
    """softmax_doverd_inputs_from_doverd_outputs:

    Returns the derivatives d C / d INPUTS from the derivatives d C / d OUTPUTS, where INPUTS and OUTPUTS are what
    goes into and comes out of the softmax function
    """

    return squeeze(((atleast_3d(doverd_outputs).repeat(d_outputs_over_d_inputs.shape[2], axis = 2)
        * d_outputs_over_d_inputs).sum(1, keepdims = True)).transpose(0, 2, 1), axis = 2)