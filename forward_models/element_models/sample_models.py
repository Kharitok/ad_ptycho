"""
Sample models for AD-based ptychography
"""
import torch.nn.functional as F
import torch.nn as nn
import torch as th
import numpy as np

# import torch.fft as th_fft
# from torch.utils.data import Dataset, DataLoader

# from propagators import grid

# th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
# th.backends.cudnn.benchmark = True


# ___________Sample models___________


class SampleComplex(th.nn.Module):
    """Sample model implemented as complex tensor"""

    def __init__(self, sample_size=None, init_sample=None):
        super().__init__()

        if (sample_size is None) and (init_sample is None):
            raise ValueError("Either sample_size or init_sample should be given")
        elif init_sample is not None:
            self.sample = nn.Parameter(th.from_numpy(init_sample).cfloat())
        else:
            self.sample = nn.Parameter(th.ones(sample_size, dtype=th.complex64))

    def forward(self):
        """Returns transfer function of the sample"""
        return self.sample


class SampleDoubleReal(th.nn.Module):
    """Sample model implemented as two real tensors tensor"""

    def __init__(self, sample_size=None, init_sample=None):
        super().__init__()

        if (sample_size is None) and (init_sample is None):
            raise ValueError("Either sample_size or init_sample should be given")
        elif init_sample is not None:
            self.sample_real = nn.Parameter(th.from_numpy(init_sample.real).float())
            self.sample_imag = nn.Parameter(th.from_numpy(init_sample.imag).float())
        else:
            self.sample_real = nn.Parameter(th.ones(sample_size, dtype=th.float32))
            self.sample_imag = nn.Parameter(th.zeros(sample_size, dtype=th.float32))

    def forward(self):
        """Returns transfer function of the sample"""
        return th.complex(self.sample_real, self.sample_imag)


class SampleRefractive(th.nn.Module):
    """Refractive sample model implemented as complex tensor"""

    def __init__(self, sample_size=None, init_sample=None):
        super().__init__()

        if (sample_size is None) and (init_sample is None):
            raise ValueError("Either sample_size or init_sample should be given")
        elif init_sample is not None:
            self.sample = nn.Parameter(th.from_numpy(init_sample).cfloat())
        else:
            self.sample = nn.Parameter(th.zeros(sample_size, dtype=th.complex64))

    def forward(self):
        """Returns transfer function of the sample"""
        return th.exp(1j * self.sample)

    def get_transmission_and_pase(self):
        """Returns transmission and phase of the sample"""
        trans = th.exp(-1 * th.imag(self.sample.detach().cpu()))
        phase = th.real(self.sample.detach().cpu())
        return (trans, phase)
    

class SampleRefractiveConstrained(th.nn.Module):
    """Refractive sample model implemented as complex tensor"""

    def __init__(self, sample_size=None, init_sample=None):
        super().__init__()

        if (sample_size is None) and (init_sample is None):
            raise ValueError("Either sample_size or init_sample should be given")
        elif init_sample is not None:
            trans = np.abs(init_sample)
            imag = np.log(np.log(trans)*-1)
            phase = np.angle(init_sample)# wrapping is here but so far we don't care
            self.sample = nn.Parameter(th.from_numpy(phase + 1j*imag).cfloat())
        else:
            self.sample = nn.Parameter(th.zeros(sample_size, dtype=th.complex64)-5j)# for having ~ 1.0 sample

    def forward(self):
        """Returns transfer function of the sample"""
        #return th.exp(1j * (th.real(self.sample) +th.exp(th.imag(self.sample))))
        return th.exp(1j * (th.real(self.sample) +1j*th.exp(th.imag(self.sample))))
    

    def get_transmission_and_pase(self):
        """Returns transmission and phase of the sample"""
        trans = th.exp(-1 * th.exp(th.imag(self.sample.detach().cpu())))
        phase = th.real(self.sample.detach().cpu())
        return (trans, phase)
    
    # torch.nn.functional.conv2d(sample[None,None,...],kernel[None,None,...],padding ='same',)
class SampleRefractiveConstrained_split(th.nn.Module):
    """Refractive sample model implemented as complex tensor"""

    def __init__(self, sample_size=None, init_sample=None):
        super().__init__()

        if (sample_size is None) and (init_sample is None):
            raise ValueError("Either sample_size or init_sample should be given")
        elif init_sample is not None:
            trans = np.abs(init_sample)
            inv_sig = -np.log(1/trans-1) 
            phase = np.angle(init_sample)# wrapping is here but so far we don't care
            self.sample_trans = nn.Parameter(th.from_numpy(inv_sig).float())
            self.sample_phase = nn.Parameter(th.from_numpy(phase).float())
        else:
            # self.sample = nn.Parameter(th.zeros(sample_size, dtype=th.complex64)-5j)# for having ~ 1.0 sample
            # self.sample_trans = nn.Parameter(th.from_numpy(imag).float())
            # self.sample_phaser = nn.Parameter(th.from_numpy(phase).float())
            raise(ValueError)

    def forward(self):
        """Returns transfer function of the sample"""
        #return th.exp(1j * (th.real(self.sample) +th.exp(th.imag(self.sample))))
        # return th.sigmoid(self.sample_trans)*th.exp(2j*th.pi*self.sample_phase)
        return th.sigmoid(self.sample_trans)*th.exp(1j*self.sample_phase)
    

    def get_transmission_and_pase(self):
        """Returns transmission and phase of the sample"""
        trans = th.exp(-1 * th.exp(self.sample_trans.detach().cpu()))
        phase = self.sample_phase.detach().cpu()
        return (trans, phase)



class SampleRefractiveConstrained_split_conv(th.nn.Module):
    """Refractive sample model with convolutional fitting"""

    def __init__(self, filter,sample_size=None, init_sample=None,):
        super().__init__()

        if (sample_size is None) and (init_sample is None):
            raise ValueError("Either sample_size or init_sample should be given")
        elif init_sample is not None:
            trans = np.abs(init_sample)
            inv_sig = -np.log(1/trans-1) 
            phase = np.angle(init_sample)# wrapping is here but so far we don't care

            self.register_buffer("sample_trans", th.from_numpy(inv_sig).float().data)
            self.register_buffer("sample_phase",th.from_numpy(phase).float().data)
            # self.sample_trans = nn.Parameter(th.from_numpy(inv_sig).float())
            # self.sample_phase = nn.Parameter(th.from_numpy(phase).float())
        else:
            # self.sample = nn.Parameter(th.zeros(sample_size, dtype=th.complex64)-5j)# for having ~ 1.0 sample
            # self.sample_trans = nn.Parameter(th.from_numpy(imag).float())
            # self.sample_phaser = nn.Parameter(th.from_numpy(phase).float())
            raise(ValueError)
        
        self.filter = nn.Parameter(th.from_numpy(filter))

    def forward(self):
        """Returns transfer function of the sample"""
        #return th.exp(1j * (th.real(self.sample) +th.exp(th.imag(self.sample))))
        # return th.sigmoid(self.sample_trans)*th.exp(2j*th.pi*self.sample_phase)
        return th.nn.functional.conv2d((th.sigmoid(self.sample_trans)*th.exp(1j*self.sample_phase))[None,None,...],self.filter[None,None,...],padding ='same')[0,0,...]
    # sample[None,None,...],kernel[None,None,...],padding ='same',

    def get_transmission_and_pase(self):
        """Returns transmission and phase of the sample"""
        trans = th.exp(-1 * th.exp(self.sample_trans.detach().cpu()))
        phase = self.sample_phase.detach().cpu()
        return (trans, phase)










def thresh(x,a=0.3):
    return (1+th.tanh(x/a))

import torch.nn as nn




def thresh(x,a=0.3):
    return (1+th.tanh(x/a))


class Sample_binary(th.nn.Module):
    """Binary sample to reconstruct diffuser"""

    def __init__(self, trans,phase,sample_size=None, init_sample=None,a_thresh =1):
        super().__init__()

        if (sample_size is None) and (init_sample is None):
            raise ValueError("Either sample_size or init_sample should be given")
        elif init_sample is not None:
            
            self.trans_max = trans
            self.phase_max = phase
            self.a = a_thresh

            self.sample_trans = nn.Parameter(th.from_numpy(init_sample).float())

            
        else:
            # self.sample = nn.Parameter(th.zeros(sample_size, dtype=th.complex64)-5j)# for having ~ 1.0 sample
            # self.sample_trans = nn.Parameter(th.from_numpy(imag).float())
            # self.sample_phaser = nn.Parameter(th.from_numpy(phase).float())
            raise(ValueError)

    def forward(self):
        """Returns transfer function of the sample"""
        #return th.exp(1j * (th.real(self.sample) +th.exp(th.imag(self.sample))))

        
        return (1 - thresh(self.sample_trans,a=self.a)/2*self.trans_max)*th.exp(1j*thresh(self.sample_trans,a=self.a)/2*self.phase_max)





class Sample_point_approximator(th.nn.Module):
    """Fits point arrays"""

    def __init__(self,bragg_angle,per_x,per_z,per_y,shift_x,shift_z,shift_y,t_z,t_x,t_y, sample_size=None, init_sample=None,):
        super().__init__()

        if (sample_size is None) and (init_sample is None):
            raise ValueError("Either sample_size or init_sample should be given")
        elif init_sample is not None:
            co_z = th.arange(init_sample.shape[0])[:,None,None]
            co_x = th.arange(init_sample.shape[1])[None,:,None]
            co_y = th.flip(th.arange(init_sample.shape[2])[None,None,:],[2])

        else:
            co_z = th.arange(sample_size[0])[:,None,None]
            co_x = th.arange(sample_size[1])[None,:,None]
            co_y = th.flip(th.arange(sample_size[2])[None,None,:],[2])
        
        
        co_z=co_z -co_z.numel()/2
        co_x=co_x -co_x.numel()/2
        co_y=co_y -co_y.numel()/2
            
        mb = th.tensor(np.radians(90)-bragg_angle)
        
        self.cz = (th.sin(mb)*co_y) + (th.cos(mb)*co_z) 
        self.cx = co_x
        self.cy = ((-th.sin(mb)*co_z)  + (th.cos(mb)*co_y) )

        self.register_buffer("coord_z", self.cz.data)
        self.register_buffer("coord_x", self.cx.data)
        self.register_buffer("coord_y", self.cy.data)
        
        
        self.per_x = nn.Parameter(th.tensor(per_x.clone().detach()).float())
        self.per_z = nn.Parameter(th.tensor(per_z.clone().detach()).float())
        self.per_y = nn.Parameter(th.tensor(per_y.clone().detach()).float())
        
        self.shift_x = nn.Parameter(th.tensor(shift_x).float())
        self.shift_z = nn.Parameter(th.tensor(shift_z).float())
        self.shift_y = nn.Parameter(th.tensor(shift_y).float())
        
        self.t_z = nn.Parameter(th.tensor(t_z).float())
        self.t_x =  nn.Parameter(th.tensor(t_x).float())
        self.t_y =  nn.Parameter(th.tensor(t_y).float())
        



    def forward(self):
        """Returns transfer function of the sample"""
        return th.sigmoid(4*(thresh(th.sin((self.coord_z-self.shift_z)*(2*th.pi/self.per_z)) -th.sigmoid(self.t_z),0.3) *
        thresh(th.sin((self.coord_x-self.shift_x)*(2*th.pi/self.per_x)) -th.sigmoid(self.t_x),0.3) *
        thresh(th.sin((self.coord_y-self.shift_y)*(2*th.pi/self.per_y)) -th.sigmoid(self.t_y),0.3))-6)
            
    



# import torch.nn.functional as F

# pt = points_t[None,None,...].clone().to(th.float32).cuda().requires_grad_(True)
# hole_th = th.tensor(hole).to(th.float32).requires_grad_(True)




# rot = th.tensor(0.0).requires_grad_(True)
# s_x,s_y = th.tensor(1.0).requires_grad_(True),th.tensor(1.0).requires_grad_(True)
# sh_x,sh_y = th.tensor(0.0).requires_grad_(True),th.tensor(0.0).requires_grad_(True)
# thicknes_max = th.tensor(1.1e-6).requires_grad_(True)



# rotation = th.stack([th.stack([th.cos(rot),-th.sin(rot),zero_t]),
#             th.stack([th.sin(rot),th.cos(rot),zero_t]),
#             uni_t])


# scale = th.stack([th.stack([s_x,zero_t,zero_t]),
#             th.stack([zero_t,s_y,zero_t]),
#             uni_t])

# shear = th.stack([th.stack([one_t,sh_x,zero_t]),
#             th.stack([sh_y,one_t,zero_t]),
#             uni_t])



# full_theta = (rotation@scale@shear)[None,0:2,...].cuda()#th.tensor([[1,0,0],[0,1,0]]).to(th.float32)[None,...]

# grid = F.affine_grid(
#             full_theta, pt.size(), align_corners=False
#         ) 

# ptr = F.grid_sample(
#             pt, grid, padding_mode="zeros", mode='bilinear', align_corners=False
#     )[0,0,...]
# plt.figure()

# plt.imshow(ptr.cpu().detach())
# plt.colorbar()


# conv = th.nn.functional.conv2d(ptr[None,None,...],hole_th[None,None,...].cuda(),padding ='same')[0,0,...]

# thick = ((th.sigmoid(conv*5)-0.5)*2)*1.1e-6
# delta,beta = 3.6420075E-05 , 2.59494573E-06#0.000103282298,  1.54670233E-05
# k = 2*np.pi/wavel

# tf = th.exp(-k*beta*thick)*th.exp(-1j*k*thick*delta)
# plt.imshow(th.angle(tf.detach().cpu()))
# plt.colorbar()


class Sample_diffuser(th.nn.Module):
    """Sample model based on the known diffuser transmission and phase"""

    def __init__(self,centers,hole_shape,thicknes_max = 1.1e-6,delta =3.6420075E-05 ,beta =  2.59494573E-06,wavel=0.137e-9):
        super().__init__()

        self.register_buffer("pt", centers[None,None,...]) 


        self.hole_shape = nn.Parameter(hole_shape.clone())
        
        self.rot = nn.Parameter(th.tensor(0.0))
        self.s_x,self.s_y = nn.Parameter(th.tensor(1.0)),nn.Parameter(th.tensor(1.0))
        self.sh_x,self.sh_y = nn.Parameter(th.tensor(0.0)),nn.Parameter(th.tensor(0.0))
        self.thicknes_max = nn.Parameter(th.tensor(thicknes_max))

        self.register_buffer("one_t", th.tensor(1.0).data)
        self.register_buffer("zero_t", th.tensor(0.0).data)
        self.register_buffer("uni_t", th.tensor([0,0,1.0]).data)

        self.delta,self.beta = delta,beta#0.000103282298,  1.54670233E-05
        self.k = 2*np.pi/wavel
 
    def forward(self):
        """Returns transfer function of the sample"""

        rotation = th.stack([th.stack([th.cos(self.rot),-th.sin(self.rot),self.zero_t]),
                    th.stack([th.sin(self.rot),th.cos(self.rot),self.zero_t]),
                    self.uni_t])


        scale = th.stack([th.stack([self.s_x,self.zero_t,self.zero_t]),
                    th.stack([self.zero_t,self.s_y,self.zero_t]),
                    self.uni_t])

        shear = th.stack([th.stack([self.one_t,self.sh_x,self.zero_t]),
                    th.stack([self.sh_y,self.one_t,self.zero_t]),
                    self.uni_t])

        full_theta = (rotation@scale@shear)[None,0:2,...]
    

        grid = F.affine_grid(
            full_theta, self.pt.size(), align_corners=False
        ) 

        ptr = F.grid_sample(
                    self.pt, grid, padding_mode="zeros", mode='bilinear', align_corners=False
            )[0,0,...]
    
        conv = th.nn.functional.conv2d(ptr[None,None,...],self.hole_shape[None,None,...].cuda(),padding ='same')[0,0,...]

        thick = ((th.sigmoid(conv*5)-0.5)*2)*self.thicknes_max    
        
        return th.exp(-self.k*self.beta*thick)*th.exp(-1j*self.k*thick*self.delta)
class SampleVariableThickness(th.nn.Module):
    """Sample model based on the constant refractive index and variable htickness"""

    pass
