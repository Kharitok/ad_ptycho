import numpy as np
import torch as th
import matplotlib.pyplot as plt
from scipy.linalg import orth
from torch.fft import fft2,ifft2

def ff(X):
    return fft2(X,norm = 'ortho')
def iff(X):
    return ifft2(X,norm = 'ortho')





def get_orth_matrix(M, N):
    Phi = th.rand(N, N)*np.exp(1j*th.rand(N, N))
    svd = th.linalg.svd (Phi)
    orth = svd[0] @ svd[2]
    return orth


size = 3
x_test = th.rand(size,size)*th.exp(1j*th.rand(size,size))


meas = th.abs(th.rand_like(x_test))



A = get_orth_matrix(size,size)
B = get_orth_matrix(size,size)
C = get_orth_matrix(size,size)



x_test.requires_grad_(True)

psi = A@B@C@x_test
I=th.abs(psi)**2

L = (  (th.sqrt(I)-th.sqrt(meas))**2   ).sum()


L.backward()


with th.no_grad():

   
    
    dI_dpsi = th.conj(psi)*(1-th.sqrt(meas)/th.abs(psi))

    m_grad = C.H@B.H@A.H@th.conj(dI_dpsi)

ad_grad = x_test.grad


print(th.allclose(th.abs(2*m_grad),th.abs(ad_grad),rtol=1e-3))
print(th.allclose(th.angle(m_grad),th.angle(ad_grad),rtol=1e-3))
print(th.abs(th.abs(2*m_grad)-th.abs(ad_grad)).sum().item())
print((th.angle(m_grad)-th.angle(ad_grad)).sum().item())






# x_test = th.rand(size,size)*th.exp(1j*th.rand(size,size))
# y_test = th.rand(size,size)*th.exp(1j*th.rand(size,size))



# th.allclose( th.conj(x_test@y_test),(th.conj(x_test)@th.conj(y_test)) )
# th.allclose( (x_test@y_test).H,( y_test.H@x_test.H) )