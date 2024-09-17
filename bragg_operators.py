import torch  as th 




def Adj_test(forward,backward,data_matrix_shape,parameters_matrix_shape):
    
    data = th.complex(th.randn(data_matrix_shape), th.randn(data_matrix_shape))
    params = th.complex(th.randn(parameters_matrix_shape), th.randn(parameters_matrix_shape))
    
    forward_pass_result = forward(params)
    bacward_pass_result = backward(data)
    
    left_side = th.sum(forward_pass_result.conj() * data)
    right_side = th.sum(params.conj() * bacward_pass_result)
    
    return left_side-right_side



