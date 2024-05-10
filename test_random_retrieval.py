import torch as th 
import matplotlib.pyplot as plt 

a = th.arange(11)
a = a-5
a

psi = (a**2)[:,None]+(a**2)[None,:]<4**2
psi = psi.cfloat()*1.0
plt.imshow(th.abs(psi))

meas_num = 30000
S = th.rand(meas_num,psi.numel())*(th.exp(1j*3.141592*th.rand(meas_num,psi.numel())))

Y = th.abs(S@psi.flatten())**2

Z = th.conj(S).T @ th.diag(Y.cfloat()) @ S

L, V = th.linalg.eig(Z)

plt.figure()
plt.imshow(th.abs(V[:,0].view(-1,11)))


print(meas_num//a.numel()**2)