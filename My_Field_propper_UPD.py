from types import CellType
import numpy as np
import numpy.fft as nfft
import matplotlib.pyplot as plt
import scipy.signal as ss

from scipy.fftpack import fftn, ifftn, fftshift, ifftshift, fftfreq
from copy import deepcopy
import scipy
import pickle
import scipy.misc
import scipy.signal
# import skimage.data
from scipy.interpolate import interp2d
from colorsys import hsv_to_rgb

#   metrical units constants

m = 1
mm = 1e-3
cm = 1e-2
um = 1e-6
nm = 1e-9


def ff(a):
    return fftshift(nfft.fft2(ifftshift(a), norm="backward"))


def iff(a):
    return fftshift(nfft.ifft2(ifftshift(a), norm="backward"))


class Field:
    def __init__(
        self,
        num=2 ** 10,
        wavel=13.5 * nm,
        size=13 * um * 2 ** 10,
        field=None,
        filename=None,
    ):
        """
        Field constructor
        default walues correspond to detector parametres:
        2^10 pixels with pixel size of 13 um
        default wavelength - 1 um
        """
        if filename is None:
            self.size = size
            self.num = num
            self.wavel = wavel
            self.cell = size / num
            self.k = 2 * np.pi / wavel
            self.F_num = None
            self.Ch_size = None

            self.field = field
            if self.field is None:
                self.field_init()

    def __repr__(self):
        self.show(mode=0, msg="")
        return f"""Field:\nSize(m):\t{self.size}\t
        \nPixel number:\t{self.num}\t
        \nWavelength(m):\t{self.wavel}\t"""

    def field_init(self):
        self.field = np.full((self.num, self.num), 1 + 0j)

        """
        initialize the field as a plane wave with a unit intensity
        """

    def __add__(self, other):
        """
        magical method for adding two fields
        """
        if (
            (self.num != other.num)
            or (self.size != other.size)
            or (self.wavel != other.wavel)
            or (self.cell != other.cell)
        ):
            raise ValueError("Fields with different parameters cannot be added")

        res = self.copy()
        res.field += other.field
        return res

    def __sub__(self, other):
        """
        magical method for substracting two fields
        """
        if (
            (self.num != other.num)
            or (self.size != other.size)
            or (self.wavel != other.wavel)
            or (self.cell != other.cell)
        ):
            raise ValueError("Fields with different parameters cannot be added")

        res = self.copy()
        res.field -= other.field
        return res

    def smooth_init(self, sigma):
        self.field *= np.random.rand(self.num, self.num)
        self.field *= scipy.ndimage.gaussian_filter(
            abs(self.field), sigma, mode="constant"
        )

    def G_H_init(self, w_0, modes_x, modes_y, z=0):

        k = self.k
        lm = self.wavel
        z_r = np.pi * w_0 ** 2 / lm
        xx, yy = self.grid()
        w = w_0 * np.sqrt(1 + (z / z_r) ** 2)

        if not (z == 0):
            R = z * (1 + (z_r / z) ** 2)
        m = len(modes_y) - 1
        n = len(modes_x) - 1

        x_herm = xx * np.sqrt(2) / w
        x_gaus = np.exp(-(xx ** 2) / w ** 2)

        y_herm = yy * np.sqrt(2) / w
        y_gaus = np.exp(-(yy ** 2) / w ** 2)

        if not (z == 0):
            P_term = np.exp(
                -1j
                * (
                    (k * z)
                    - ((1 + n + m) * np.arctan(z / z_r))
                    + ((k * (xx ** 2 + yy ** 2)) / (2 * R))
                )
            )
        else:
            P_term = np.exp(-1j * ((k * z) - ((1 + n + m) * np.arctan(z / z_r))))

        self.field = (
            np.polynomial.hermite.hermval(x_herm, modes_x)
            * x_gaus
            * np.polynomial.hermite.hermval(y_herm, modes_y)
            * y_gaus
            * P_term
        )

    def G_L_init(self, w_0, p, l, z=0):

        xx, yy = self.grid()
        ro_ = np.sqrt(xx ** 2 + yy ** 2)
        phi_ = np.arctan2(yy, xx)
        k = self.k
        lm = self.wavel

        p = len(p) - 1
        l = len(l) - 1

        z_r = np.pi * w_0 ** 2 / lm
        w_z = w_0 * np.sqrt(1 + (z / z_r) ** 2)
        psi_z = (np.abs(l) + 2 * p + 1) * (np.arctan2(z, z_r))
        if z != 0:
            R_z = z * (1 + (z_r / z) ** 2)

        T1 = w_0 / w_z
        T2 = (ro_ * np.sqrt(2) / w_z) ** np.abs(l)
        T3 = np.exp(-(ro_ ** 2) / (w_z ** 2))
        arg_ = 2 * ro_ ** 2 / (w_z ** 2)
        L = scipy.special.laguerre(p, l)
        if z != 0:
            phase_ = (
                np.exp(-1j * k * ro_ ** 2 / (2 * R_z))
                * np.exp(-1j * l * phi_)
                * np.exp(1j * psi_z)
            )
        else:
            phase_ = np.exp(-1j * l * phi_) * np.exp(1j * psi_z)

        self.field = T1 * T2 * T3 * L(arg_) * phase_  # *(np.cos(l*phi_)/np.sin(l*phi_))

    def update_cell(self, new_cell):
        self.cell = new_cell
        self.size = self.cell * self.num

    def update_size(self, new_size):
        self.size = new_size
        self.cell = self.size / self.num

    def normalize(self, inplace=True):
        if inplace:
            self.field = self.field / (np.max(np.abs(self.field)))
        else:
            res = self.copy()
            res.field = self.field / (np.max(np.abs(self.field)))
            return res

    def copy(self):
        """
        return deep copy of a field
        """
        return deepcopy(self)

    def phi(self, u=False):
        """
        returns phase of a field
        """
        if u:
            return np.unwrap(np.angle((self.field)))
        return np.angle((self.field))

    def int_(self):
        """
        returns intensity of a field
        """
        return np.real(self.field * np.conj(self.field))

    def grid(self, sparse=True):
        """
        return grid of the field as a meshgrid
        """
        dx, dy = self.cell, self.cell
        Nx, Ny = self.num, self.num

        # grid = np.linspace(start=(-Nx / 2), stop=(Nx / 2 - 1), num=Nx)
        grid = (np.arange(0,Nx,1)-Nx//2)

        Xn = grid * dx
        Yn = grid * dy

        xx, yy = np.meshgrid(Xn, Yn, sparse=sparse)

        return (xx, yy)

    def freq_grid(self):
        """
        Returns Fourrier frequencies grid as a meshgrid
        """

        fx = np.fft.ifftshift(np.fft.fftfreq(n= self.num,d = self.cell))
        fxx, fyy = np.meshgrid(fx, fx, sparse=True)

        return(fxx,fyy)

    def fourrier_scaled_grid(self, z,):
        """
        Returns coordinate grid in the target plane acourding to the fourrier scaling 
        dx2 = lm*z/(N*dx)
        """

        lm = self.wavel
        grid2 = (np.arange(0,self.num,1)-self.num//2)

        dx2 = lm*np.abs(z)/(self.num*self.cell)
        dy2 = dx2

        x2,y2 = np.meshgrid(grid2 * dx2, grid2 * dy2, sparse=True)
        return (x2,y2)

    def reciprocal_grid(self, sparse=True):
        """
        return reciprocal grid of the field as a meshgrid
        """
        dx, dy = self.cell, self.cell
        Nx, Ny = self.num, self.num
        dkx, dky = 2 * np.pi / (Nx * dx), 2 * np.pi / (Ny * dy)
        grid = np.linspace(start=(-Nx / 2), stop=(Nx / 2 - 1), num=Nx)

        Kxn = grid * dkx
        Kyn = grid * dky

        kxx, kyy = np.meshgrid(Kxn, Kyn, sparse=sparse)

        return (kxx, kyy)

    def save(self, Filename):
        with open(Filename, "wb") as handle:
            pickle.dump(self, handle)

    def show(
        self,
        units=True,
        unwrap=False,
        norm=True,
        mode="Full",
        filename=None,
        msg="",
        disp_c=False,
    ):
        """
        Plot field
        units - use pxels or actual size
        mode - determine either  plot only intensity(='Full')
        or intensity phase and central cuts
        norm - normalize intensity of the field
        unwrap - unwrap phase of the field
        """

        Style_plot(
            self,
            units=units,
            unwrap=unwrap,
            norm=norm,
            mode=mode,
            msg=msg,
            disp_c=disp_c,
        )
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def Rescale(self, dx_new, inplace=False, new_size=None, x0=0, y0=0, kind="linear"):
        """ Rescale square cut from the field  to new pix size pix num"""
        # new grid
        if new_size is None:
            new_size = self.size

        Nx_new = new_size / dx_new
        grid_new = np.linspace(start=(-Nx_new / 2), stop=(Nx_new / 2 - 1), num=Nx_new)
        X_new = (grid_new * dx_new) - x0
        Y_new = (grid_new * dx_new) - y0
        xx_new, yy_new = np.meshgrid(X_new, Y_new, sparse=True)

        # old grid
        # xx_old, yy_old = self.grid(sparse=True)
        # select part of the data for interpolation

        # interp
        real_new = interp2d(
            x=self.grid(sparse=True)[0],
            y=self.grid(sparse=True)[1],
            z=np.real(self.field),
            bounds_error=False,
            fill_value=0,
            kind=kind,
        )
        F_real = real_new(xx_new.flatten(), yy_new.flatten())
        img_new = interp2d(
            x=self.grid(sparse=True)[0],
            y=self.grid(sparse=True)[1],
            z=np.imag(self.field),
            bounds_error=False,
            fill_value=0,
            kind=kind,
        )
        F_imag = img_new(xx_new.flatten(), xx_new.flatten())
        Field_new = F_real + 1j * F_imag

        if not inplace:
            New_fied = Field(
                num=Nx_new, wavel=self.wavel, size=Nx_new * dx_new, field=Field_new
            )
            return New_fied
        else:
            self.field = Field_new
            self.num = Nx_new
            self.cell = dx_new
            self.size = self.cell * self.num

    def F_update(self, z, inplace=True):
        """
        Update Fresnel number during propofation
        """
        if inplace:
            self.F_num = Count_fresnel_number(D=self.Ch_size, z=z, lm=self.wavel)
            return self.F_num
        else:
            return Count_fresnel_number(D=self.Ch_size, z=z, lm=self.wavel)

    def c_a(self, r=1 * mm, x=0, y=0):
        """
        Apply circular aperture;
        r - radius   of  aperture
        x,y- center coordinates
        """

        xx, yy = self.grid()

        F = np.where((((xx - x) ** 2 + (yy - y) ** 2) < r ** 2), 1, 0)
        self.field *= F

    def c_s(self, r=1 * mm, x=0, y=0):
        """
        Apply circular aperture;
        r - radius   of  aperture
        x,y- center coordinates
        """

        xx, yy = self.grid()

        F = np.where((((xx - x) ** 2 + (yy - y) ** 2) < r ** 2), 0, 1)
        self.field *= F

    def r_s(self, w=1 * mm, h=2 * mm, x=0, y=0):
        """
        Apply rectangular screen;
        w,h - width  and height of screen
        x,y- center coordinates
        """
        xx, yy = self.grid()

        F = np.where(((abs(xx - x) < w / 2) & (abs(yy - y) < h / 2)), 0, 1)
        self.field *= F

    def r_a(self, w=1 * mm, h=2 * mm, x=0, y=0):
        """
        Apply rectangular aperture;
        w,h - width  and height of aperture
        x,y- center coordinates
        """
        xx, yy = self.grid()

        F = np.where(((abs(xx - x) < w / 2) & (abs(yy - y) < h / 2)), 1, 0)
        self.field *= F

    def g_a(self, iw=1 * mm, x=0, y=0, ci=1):
        """
        Apply gaussian aperture;
        iw - 1/e intensity width
        x,y- center coordinates
        ci - center transmission
        """

        dx, dy = self.cell, self.cell
        Nx, Ny = self.num, self.num
        dkx, dky = 2 * np.pi / (Nx * dx), 2 * np.pi / (Ny * dy)
        grid = np.linspace(start=(-Nx / 2), stop=(Nx / 2 - 1), num=Nx)

        Xn = grid * dx
        Yn = grid * dy
        Kxn = grid * dkx
        Kyn = grid * dky
        xx, yy = np.meshgrid(Xn, Yn, sparse=True)
        kxx, kyy = np.meshgrid(Kxn, Kyn, sparse=True)

        F = (1 + 0j) * (
            np.sqrt(ci) * np.exp(-(((xx - x) ** 2 + (yy + y) ** 2) / (2 * iw ** 2)))
        )
        self.field *= F

    def g_s(self, iw=1 * mm, x=0, y=0, ci=1):
        """
        Apply gaussian screen;
        iw - 1/e intensity width
        x,y- center coordinates
        ci - center transmission
        """
        dx, dy = self.cell, self.cell
        Nx, Ny = self.num, self.num
        dkx, dky = 2 * np.pi / (Nx * dx), 2 * np.pi / (Ny * dy)
        grid = np.linspace(start=(-Nx / 2), stop=(Nx / 2 - 1), num=Nx)

        Xn = grid * dx
        Yn = grid * dy
        Kxn = grid * dkx
        Kyn = grid * dky
        xx, yy = np.meshgrid(Xn, Yn, sparse=True)
        kxx, kyy = np.meshgrid(Kxn, Kyn, sparse=True)

        F = (1 + 0j) * (
            np.sqrt(
                1
                - (1 - ci) * np.exp(-(((xx - x) ** 2 + (yy + y) ** 2) / (2 * iw ** 2)))
            )
        )
        self.field *= F

    def z_p(self, f, r=None, smooth=True):  # dx <=lm*f/(2r)
        dx, dy = self.cell, self.cell
        Nx, Ny = self.num, self.num
        dkx, dky = 2 * np.pi / (Nx * dx), 2 * np.pi / (Ny * dy)
        grid = np.linspace(start=(-Nx / 2), stop=(Nx / 2 - 1), num=Nx)

        Xn = grid * dx
        Yn = grid * dy
        Kxn = grid * dkx
        Kyn = grid * dky
        xx, yy = np.meshgrid(Xn, Yn, sparse=True)
        kxx, kyy = np.meshgrid(Kxn, Kyn, sparse=True)

        if smooth:
            zp_trans = 1 / 2 * (1 + np.cos((self.k / (2 * f)) * (xx ** 2 + yy ** 2)))
        else:
            zp_trans = (
                1 / 2 * (1 + np.sign(np.cos((self.k / (2 * f)) * (xx ** 2 + yy ** 2))))
            )

        self.field = self.field * zp_trans
        if not (r is None):
            self.c_a(r)

    def sum(self, F):
        self.field += F.field

    def custom_aperture(self, f):
        """
        Apply castom aperture
        """
        F = (1 + 0j) * f
        self.field *= F

    def sph_init(self, z0, x0=0, y0=0):
        """
        initialize field as a field radiating from point source

        z0,x0,y0 - coordinates of a source

        """
        dx, dy = self.cell, self.cell
        Nx, Ny = self.num, self.num
        dkx, dky = 2 * np.pi / (Nx * dx), 2 * np.pi / (Ny * dy)
        grid = np.linspace(start=(-Nx / 2), stop=(Nx / 2 - 1), num=Nx)

        Xn = grid * dx
        Yn = grid * dy
        Kxn = grid * dkx
        Kyn = grid * dky
        xx, yy = np.meshgrid(Xn, Yn, sparse=True)
        kxx, kyy = np.meshgrid(Kxn, Kyn, sparse=True)

        R = z0
        F = (
            np.exp(1j * self.k * ((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * R))
            * np.exp(1j * self.k * R)
            / (R)
        )

        self.field *= F

    def lens(self, f, r=None, x=0, y=0):
        """
        Apply  phase mask accourding;
        x,y - center coordinates,
        f  - focal distance +conv - div
        """

        xx, yy = self.grid()

        F = np.exp((-1j * self.k) * (((xx - x) ** 2 + (yy - y) ** 2) / (2 * f)))

        self.field *= F
        if not (r is None):
            self.c_a(r)

    def x_lens(self, f, r=None, x=0):
        """
        Apply  phase mask accourding;
        x - center coordinates,
        f  - focal distance
        """

        xx, yy = self.grid()

        F = np.exp((-1j * self.k) * (((xx - x) ** 2) / (2 * f)))

        self.field *= F
        if not (r is None):
            self.c_a(r)

    def y_lens(self, f, r=None, y=0):
        """
        Apply  phase mask accourding;
        x - center coordinates,
        f  - focal distance
        """

        xx, yy = self.grid()

        F = np.exp((-1j * self.k) * (((yy - y) ** 2) / (2 * f)))

        self.field *= F
        if not (r is None):
            self.c_a(r)

    #   convolutional routines

    def tilt(self, alpha=0, theta=0):
        """
        Apply  phase mask accourding: T(x y) = exp (jk *(x cos theta  y sin theta)
        alpha - tilt angle
        theta - rotation angle (x axis 0) ^| -90, _| 90
        """
        xx, yy = self.grid()

        F = np.exp(
            1j * self.k * (xx * np.cos(theta) + yy * np.sin(theta)) * np.tan(alpha)
        )
        self.field *= F

    def convolve(self, h):
        self.field = ss.convolve(self.field, h, mode="same")

    def Cconvolve(self, h):
        self.field = fftshift(ifftn(fftn(self.field) * fftn(h)))

    def Fconvolve(self, fh):
        self.field = fftshift((fftn(self.field) * fh))

    def Fr_scale(self, f, z):
        sc = 1
        vec = 1
        if f < 0:
            sc = -1 * sc
            f = -1 * f

        if z < 0:
            sc = -1 * sc
            z = -1 * z
            vec = -1

        M = (f + z) / f

        if sc == -1:
            M = 1 / M
        if vec == -1:
            z = -z

        print(f" {M} - M ; old cell = {self.cell}; old distance = {z} ")
        self.cell *= M
        self.field /= M
        self.size = self.cell * self.num
        print(f" new cell = {self.cell}; new distance = {z/M} ")
        return (z / M, M)

    def Fr_descale(self, M):
        self.cell /= M
        self.field *= M
        self.size = self.cell * self.num

    def FIR_prop_UPD(self, z, inplace=False,):
        """
        Fresnel impulse responce propagation 
        better for longer distances
        Δx <= λz/L
        """

        k= self.k
        lm=self.wavel
        x,y =self.grid()
        dx= self.cell
        #np.exp(1j*k*z)
        h = (1/(1j*lm*z)) * np.exp(((1j*k)/(2*z))*(x**2+y**2))
        H = ff(h)*dx**2

        if inplace:
            self.field = iff(ff(self.field) * H)
        else:
            res = self.copy()
            res.field = iff(ff(self.field) * H)
            return res



    def FTF_prop_UPD(self, z, inplace=False,
    ):
        """
        Fresnel transfer function responce propagation  
        better for shorter  distances
        Δx >= λz/L
        """
        
        k= self.k
        lm = self.wavel
        fx,fy = self.freq_grid()
#         (np.exp(1j*k*z))*
        H = np.exp((-1j*np.pi*lm*z)*(fx**2 +fy**2))

        if inplace:
            self.field = iff(ff(self.field) * H)
        else:
            res = self.copy()
            res.field = iff(ff(self.field) * H)
            return res

#     def FST_prop_UPD(self, z, inplace=False,
#     ):
#         """
#         Fresnel single  transform propagation  
#         """

#         k= self.k
#         lm = self.wavel
#         dx2 = lm*np.abs(z)/(self.num*self.cell)
#         x1,y1 = self.grid()

#         x2, y2 = self.fourrier_scaled_grid(z)

#         mul1 = (np.exp(1j*k*z)/(1j*z*lm))*np.exp((1j*k/(2*z))*(x2**2+y2**2))
#         mul2 = np.exp((1j*k/(2*z))*(x1**2+y1**2))

#         if inplace:
#             if z >0:
#                 self.field = mul1*ff(self.field * mul2)
#             else: 
#                 self.field = mul1*iff(self.field * mul2)
#             self.size = dx2 * self.num
#             self.cell = dx2

#         else:
#             res = self.copy()
#             if z >0:
#                 res.field = mul1*ff(self.field * mul2)
#             else: 
#                 res.field = mul1*iff(self.field * mul2)
#             res.size = dx2 * self.num
#             res.cell = dx2
#             return res
    def FST_prop_UPD(self, z, inplace=False,
    ):
        """
        Fresnel single  transform propagation  
        """

        k= self.k
        lm = self.wavel
        dx2 = lm*np.abs(z)/(self.num*self.cell)
        x1,y1 = self.grid()

        x2, y2 = self.fourrier_scaled_grid(z)

        mul1 = (np.exp(1j*k*z)/(1j*z*lm))*np.exp((1j*k/(2*z))*(x2**2+y2**2))*(0.5*dx2)
        mul2 = np.exp((1j*k/(2*z))*(x1**2+y1**2))

        if inplace:
            if z >0:
                self.field = mul1*ff(self.field * mul2)/self.num
            else: 
                self.field = mul1*iff(self.field * mul2)*self.num
            self.size = dx2 * self.num
            self.cell = dx2

        else:
            res = self.copy()
            if z >0:
                res.field = mul1*ff(self.field * mul2)/self.num
            else: 
                res.field = mul1*iff(self.field * mul2)*self.num
            res.size = dx2 * self.num
            res.cell = dx2
            return res

    def FST_prop_UPD_int_conserve(self, z, inplace=False,
    ):
        """
        Fresnel single  transform propagation  
        """

        k= self.k
        lm = self.wavel
        dx2 = lm*np.abs(z)/(self.num*self.cell)
        x1,y1 = self.grid()

        x2, y2 = self.fourrier_scaled_grid(z)

        mul1 = np.exp((1j*k/(2*z))*(x2**2+y2**2))
        mul2 = np.exp((1j*k/(2*z))*(x1**2+y1**2))

        if inplace:
            if z >0:
                self.field = mul1*ff(self.field * mul2)/self.num
            else: 
                self.field = mul1*iff(self.field * mul2)*self.num
            self.size = dx2 * self.num
            self.cell = dx2

        else:
            res = self.copy()
            if z >0:
                res.field = mul1*ff(self.field * mul2)/self.num
            else: 
                res.field = mul1*iff(self.field * mul2)*self.num
            res.size = dx2 * self.num
            res.cell = dx2
            return res


    def RSIR_prop_UPD(self, z, inplace=False,):
        """
        Rayleigh–Sommerfeld impulse responce propagation 
        better for longer distances
        Δx <= λz/L
        """

        k= self.k
        lm=self.wavel
        x,y =self.grid()
        dx = self.cell

        h = (z/(1j*lm)) * ((np.exp(1j*k*np.sqrt(z**2+x**2+y**2)))/(z**2+x**2+y**2))
        H = ff(h)*dx**2

        if inplace:
            self.field = iff(ff(self.field) * H)
        else:
            res = self.copy()
            res.field = iff(ff(self.field) * H)
            return res



    def RSTF_prop_UPD(self, z, inplace=False,
    ):
        """
        Rayleigh–Sommerfeld transfer function responce propagation  
        better for shorter  distances
        Δx >= λz/L
        """
        
        k= self.k
        lm = self.wavel
        fx,fy = self.freq_grid()

        H = np.exp(2j*np.pi*z*np.sqrt(1- (lm*fx )**2 - (lm*fy)**2)/lm)

        if inplace:
            self.field = iff(ff(self.field) * H)
        else:
            res = self.copy()
            res.field = iff(ff(self.field) * H)
            return res
        
        
    def FRA_prop_UPD(self, z, inplace=False,
    ):
        """
        Fraunhoffer propagation  
        """
        lm = self.wavel
        k = self.k
        x2,y2 = self.fourrier_scaled_grid(z)
        dx2 = lm*np.abs(z)/(self.num*self.cell)

        mul1 = 1/(1j*lm*z) * np.exp((1j*k/(2*z))*(x2**2+y2**2))* (0.5*dx2)#normalization

        if inplace:
            if z>0:

                self.field = mul1*ff(self.field)/self.num
            else:
                self.field = mul1*iff(self.field)*self.num

            self.size = dx2 * self.num
            self.cell = dx2
        else:
            res = self.copy()
            if z>0:
                res.field = mul1*ff(self.field)/self.num
            else:
                res.field = mul1*iff(self.field)*self.num

            res.size = dx2 * self.num
            res.cell = dx2
            return res


#     def FRA_prop_UPD(self, z, inplace=False,
#     ):
#         """
#         Fraunhoffer propagation  
#         """
#         lm = self.wavel
#         k = self.k
#         x2,y2 = self.fourrier_scaled_grid(z)
#         dx2 = lm*np.abs(z)/(self.num*self.cell)

#         mul1 = 1/(1j*lm*z) * np.exp((1j*k/(2*z))*(x2**2+y2**2))

#         if inplace:
#             self.field = mul1*ff(self.field)*self.cell**2
#             self.size = dx2 * self.num
#             self.cell = dx2
#         else:
#             res = self.copy()
#             res.field = mul1*ff(self.field)*self.cell**2
#             res.size = dx2 * self.num
#             res.cell = dx2
#             return res

    def FRA_prop_INT_UPD(self, z, inplace=False,
    ):
        """
        Fraunhoffer propagation  
        """
        lm = self.wavel
        k = self.k
        x2,y2 = self.fourrier_scaled_grid(z)
        dx2 = lm*np.abs(z)/(self.num*self.cell)


        if inplace:

            if z >0:
                self.field = ff(self.field)/self.num
            else:
                self.field = iff(self.field)*self.num
            self.size = dx2 * self.num
            self.cell = dx2
        else:
            res = self.copy()
            if z >0:
                res.field = ff(self.field)/self.num
            else:
                res.field = iff(self.field)*self.num
            res.size = dx2 * self.num
            res.cell = dx2
            return res
    

    def FIR_prop(
        self, z, inplace=False,
    ):
        """
        Fresnel impulse responce propagation  Psi2 = F^-1{F{Psi1}*F{h}}
        better for far distances
        Δx <= λz/L
        """
        k = self.k
        dx = self.size / self.num
        dy = dx
        Xn = np.linspace(
            -self.num / 2 * dx, self.num / 2 * dx, num=self.num, endpoint=False
        )
        Yn = np.linspace(
            -self.num / 2 * dy, self.num / 2 * dy, num=self.num, endpoint=False
        )
        xx, yy = np.meshgrid(Xn, Yn, sparse=True)

        h = np.exp((1j * k * z) + (1j * k / (2 * z) * (xx ** 2 + yy ** 2)))

        if inplace:
            self.field = iff(ff(self.field) * ff(h))
        else:
            res = self.copy()
            res.field = iff(ff(self.field) * ff(h))
            return res

    def FTF_prop(
        self, z, inplace=False,
    ):
        """
        Fresnel transfer function responce propagation  Psi2 = F^-1{F{Psi1}*H}
        better for small distances
        Δx >= λz/L
        """
        k = self.k

        fx = fftshift(fftfreq(self.field.shape[0], self.size / self.num))
        fy = fftshift(fftfreq(self.field.shape[0], self.size / self.num))
        fxx, fyy = np.meshgrid(fx, fy, sparse=True)

        H = np.exp(1j * k * z - 1j * np.pi * self.wavel * z * (fxx ** 2 + fyy ** 2))

        if inplace:
            self.field = iff(ff(self.field) * H)
        else:
            res = self.copy()
            res.field = iff(ff(self.field) * H)
            return res

    def FST_prop(self, z, inplace=False, explicit_coord=False):
        """
        Fresnel single  transform propagation
        Psi2_x2  = e^{jkz}/jkz * e^{jk(x_2^2+y_2^2)/2z}*F{Psi1_x1 * e^{jk(x_1^2+y_1^2)/2z}}
        """
        k = self.k
        lm = self.wavel
        dx1 = self.size / self.num
        dy1 = dx1
        dx2 = (self.wavel * np.abs(z)) / (self.num * dx1)
        dy2 = dx2
        Nx = self.num
        Ny = Nx

        Xn1 = np.linspace(-Nx / 2 * dx1, Nx / 2 * dx1, num=self.num, endpoint=False)
        Yn1 = np.linspace(-Ny / 2 * dy1, Ny / 2 * dy1, num=self.num, endpoint=False)
        xx1, yy1 = np.meshgrid(Xn1, Yn1, sparse=True)

        Xn2 = np.linspace(-Nx / 2 * dx2, Nx / 2 * dx2, num=self.num, endpoint=False)
        Yn2 = np.linspace(-Ny / 2 * dy2, Ny / 2 * dy2, num=self.num, endpoint=False)
        xx2, yy2 = np.meshgrid(Xn2, Yn2, sparse=True)

        Field_ = (
            np.sqrt((dx1 * dy1) / (dx2 * dy2))
            * np.exp(1j * k * z)
            * np.exp((1j * k * (xx2 ** 2 + yy2 ** 2)) / (2 * z))
            * ff(self.field * np.exp((1j * k * (xx1 ** 2 + yy1 ** 2)) / (2 * z)))
        )

        if z < 0:
            Field_ = Flip(Field_)

        if explicit_coord:
            print(
                f"Grid pixel size rescaled from {dx1} to {dx2};\n Corresponding grid sizes: {dx1*self.num}, {dx2*self.num}"
            )

        if inplace:
            self.field = Field_
            self.size = dx2 * self.num
            self.cell = self.size / self.num
        else:
            res = self.copy()
            res.field = Field_
            res.size = dx2 * res.num
            res.cell = res.size / res.num
            return res

    def FRA_prop(self, z, inplace=False):

        k = self.k
        lm = self.wavel
        dx1 = self.size / self.num
        dy1 = dx1
        dx2 = (self.wavel * np.abs(z)) / (self.num * dx1)
        dy2 = dx2
        Nx = self.num
        Ny = Nx

        Xn1 = np.linspace(-Nx / 2 * dx1, Nx / 2 * dx1, num=self.num, endpoint=False)
        Yn1 = np.linspace(-Ny / 2 * dy1, Ny / 2 * dy1, num=self.num, endpoint=False)
        xx1, yy1 = np.meshgrid(Xn1, Yn1, sparse=True)

        Xn2 = np.linspace(-Nx / 2 * dx2, Nx / 2 * dx2, num=self.num, endpoint=False)
        Yn2 = np.linspace(-Ny / 2 * dy2, Ny / 2 * dy2, num=self.num, endpoint=False)
        xx2, yy2 = np.meshgrid(Xn2, Yn2, sparse=True)

        c = 1 / (1j * lm * z) * np.exp(1j * k / (2 * z) * (xx2 ** 2 + yy2 ** 2))

        if inplace:
            self.field = (c * ff(self.field)) * dx1 ** 2

            self.cell = dx2
            self.size = self.cell * self.num

        else:

            res = self.copy()
            res.field = (c * ff(self.field)) * dx1 ** 2
            res.cell = dx2
            res.size = res.cell * res.num
            return res

    def Fra_T(self, z, F_up=False, inplace=False, f=None):
        if not (f is None):
            z, M = self.Fr_scale(f, z)

        dx, dy = self.cell, self.cell
        Nx, Ny = self.num, self.num
        dkx, dky = 2 * np.pi / (Nx * dx), 2 * np.pi / (Ny * dy)
        dx_sc = (self.wavel * z) / (Nx * dx)
        dy_sc = (self.wavel * z) / (Nx * dy)
        grid = np.linspace(start=(-Nx / 2), stop=(Nx / 2 - 1), num=Nx)

        Xn = grid * dx
        Yn = grid * dy
        Kxn = grid * dkx
        Kyn = grid * dky

        Xn_sc = grid * dx_sc
        Yn_sc = grid * dy_sc

        xx, yy = np.meshgrid(Xn, Yn, sparse=True)
        kxx, kyy = np.meshgrid(Kxn, Kyn, sparse=True)
        xx_sc, yy_sc = np.meshgrid(Xn_sc, Yn_sc, sparse=True)

        if inplace:

            self.field = (
                -1j
                * np.sqrt((dx * dy) / (dx_sc * dy_sc))
                * np.exp(1j * self.k * z)
                * np.exp((1j * self.k * (xx_sc ** 2 + yy_sc ** 2)) / (2 * z))
                * ff(self.field)
            )

            self.cell = dx_sc
            self.size = self.cell * self.num
        else:
            # print(xx.shape)
            # print(xx.shape)
            # print(xx_sc.shape)

            res = self.copy()
            # print(res.field.shape)
            res.field = (
                -1j
                * np.sqrt((dx * dy) / (dx_sc * dy_sc))
                * np.exp(1j * self.k * z)
                * np.exp((1j * self.k * (xx_sc ** 2 + yy_sc ** 2)) / (2 * z))
                * ff(self.field)
            )
            res.cell = dx_sc
            res.size = res.cell * res.num

            if not (f is None):
                self.Fr_descale(M)
            return res

    def CorProp(self):
        Fraun_d = self.Ch_size ** 2 / (2 * self.wavel)
        ro = np.sqrt(2 * self.size ** 2)
        Fres_d = (1 / 2 * ro / self.wavel) ** (4 / 3) * self.wavel
        print("Fra - {Fraun_d}\n Fre - {Fres_d}")

    def PropBorders(self):
        print("crit condition - z = {(self.num*self.cell**2)/self.wavel}")

    def Q_Prop(self, z, inplace=False):
        dx, dy = self.cell, self.cell
        Nx, Ny = self.num, self.num
        dkx, dky = 2 * np.pi / (Nx * dx), 2 * np.pi / (Ny * dy)
        dx_sc = (self.wavel * z) / (Nx * dx)
        dy_sc = (self.wavel * z) / (Nx * dy)
        grid = np.linspace(start=(-Nx / 2), stop=(Nx / 2 - 1), num=Nx)

        Xn = grid * dx
        Yn = grid * dy
        Kxn = grid * dkx
        Kyn = grid * dky

        Xn_sc = (Xn / dx) * dx_sc
        Yn_sc = (Yn / dy) * dy_sc

        xx, yy = np.meshgrid(Xn, Yn, sparse=True)
        kxx, kyy = np.meshgrid(Kxn, Kyn, sparse=True)
        xx_sc, yy_sc = np.meshgrid(Xn_sc, Yn_sc, sparse=True)

        if inplace:

            self.field = (
                -1j
                / (self.wavel * z)
                * np.exp((2 * np.pi * 1j * z) / (self.wavel))
                * np.exp((1j * np.pi * (xx_sc ** 2 + yy_sc ** 2)) / (self.wavel * z))
                * ff(
                    (
                        np.exp(
                            (1j * np.pi * (xx_sc ** 2 + yy_sc ** 2)) / (self.wavel * z)
                        )
                    )
                    * (self.field)
                )
            )

        else:
            print(xx.shape)
            print(xx.shape)
            print(xx_sc.shape)

            res = self.copy()
            print(res.field.shape)
            res.field = (
                -1j
                / (self.wavel * z)
                * np.exp((2 * np.pi * 1j * z) / (self.wavel))
                * np.exp((1j * np.pi * (xx_sc ** 2 + yy_sc ** 2)) / (self.wavel * z))
                * ff(
                    (
                        np.exp(
                            (1j * np.pi * (xx_sc ** 2 + yy_sc ** 2)) / (self.wavel * z)
                        )
                    )
                    * (self.field)
                )
            )
            return res


#   functions


def shift(ry, rx, img, axis=None):
    if axis is None:
        rx = -rx
        s_x = img.shape[0]
        s_y = img.shape[1]

        before_x = (rx >= 0) * abs(rx)
        after_x = (rx <= 0) * abs(rx)
        before_y = (ry >= 0) * abs(ry)
        after_y = (ry <= 0) * abs(ry)

        x_l_b = 0 - (rx < 0) * img.shape[0]
        x_m_b = img.shape[0] + (rx < 0) * abs(rx)
        y_l_b = 0 - (ry < 0) * img.shape[0]
        y_m_b = img.shape[0] + (ry < 0) * abs(ry)

        shifted = np.pad(
            array=img,
            pad_width=((before_x, after_x), (before_y, after_y)),
            mode="constant",
            constant_values=0 + 0j,
        )
        return shifted[x_l_b:x_m_b, y_l_b:y_m_b]
    else:
        rx = -rx
        s_x = img.shape[axis[0]]
        s_y = img.shape[axis[1]]

        before_x = (rx >= 0) * abs(rx)
        after_x = (rx <= 0) * abs(rx)
        before_y = (ry >= 0) * abs(ry)
        after_y = (ry <= 0) * abs(ry)

        x_l_b = 0 - (rx < 0) * img.shape[axis[0]]
        x_m_b = img.shape[axis[0]] + (rx < 0) * abs(rx)
        y_l_b = 0 - (ry < 0) * img.shape[axis[0]]
        y_m_b = img.shape[axis[0]] + (ry < 0) * abs(ry)

        shifted = np.pad(
            array=img,
            pad_width=[(0, 0), (before_x, after_x), (before_y, after_y)],
            mode="constant",
            constant_values=0 + 0j,
        )
        return shifted[:, x_l_b:x_m_b, y_l_b:y_m_b]


def Flip(field):
    return shift(1, -1, np.flipud(np.fliplr(field)))


def Colorize(z):
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1, 1, 1)
    c[np.isnan(z)] = (0, 0, 0)
    c[np.abs(z) == 0] = (0, 0, 0)
    idx = ~(np.isinf(z) + np.isnan(z) + (np.abs(z) == 0))
    B = np.abs(z[idx]) / np.max(np.abs(z[idx]))
    A = (np.angle(z[idx]) + np.pi) / (2 * np.pi)
    c[idx] = [hsv_to_rgb(a, b, b) for a, b in zip(A, B)]
    return c


def load_field(Filename):
    with open(Filename, "rb") as handle:
        loaded_field = pickle.load(handle)
    return loaded_field


def Count_fresnel_number(D, lm, z):
    """
    routine for counting F number
    """
    fn = (D ** 2) / (lm * z)

    if fn >= 10:
        return (fn, "near")
    elif fn <= 0.1:
        return (fn, "far")
    else:
        return (fn, "border")


def Style_plot(
    Fild, units=True, unwrap=False, norm=True, mode="Full", msg=None, disp_c=False
):
    """
    routine for field plotting

    units - use pxels or actual size

    mode - determine either  plot only intensity(='Full')
    or intensity phase and central cuts



    norm - normalize intensity of the field

    unwrap - unwrap phase of the field


    """

    if mode != "Full":
        F = Fild.copy()

        if norm:
            F.field = F.field / np.max(np.abs(F.field))

        if not disp_c:
            I1 = F.int_()
        else:
            I1 = Colorize(F.field)

        if units:
            dx, dy = F.cell, F.cell
            Nx, Ny = F.num, F.num

            Xn = np.linspace(-Nx / 2 * dx, Nx / 2 * dx, num=F.num) / mm
            Yn = np.linspace(-Ny / 2 * dy, Ny / 2 * dy, num=F.num) / mm

            im3 = plt.imshow(I1, extent=[Xn[0], Xn[-1], Yn[0], Yn[-1]])
            plt.colorbar(im3)
            plt.xlabel("mm")
            plt.ylabel("mm")
            plt.title("Intensity")

            if msg is None:
                if Fild.F_num is None:
                    plt.figtext(
                        0.05, 0.95, "Not propogated yet", fontsize=18, color="black"
                    )
                else:
                    plt.figtext(
                        0.05,
                        0.95,
                        f"Propogated with F number = {Fild.F_num}",
                        fontsize=18,
                        color="black",
                    )
            else:
                plt.figtext(0.05, 0.95, msg, fontsize=18, color="black")

        else:

            im3 = plt.imshow(I1)
            plt.colorbar(im3)
            plt.title("Intensity")
            plt.xlabel("pixels")
            plt.ylabel("pixels")

        return

    fig = plt.figure(figsize=(15, 9.5))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    F = Fild.copy()

    if norm:
        F.field = F.field / np.max(np.abs(F.field))

    if not disp_c:
        I1 = F.int_()
    else:
        I1 = Colorize(F.field)

    if unwrap:
        Phi1 = F.phi(u=True)
    else:
        Phi1 = F.phi()

    if units:
        dx, dy = F.cell, F.cell
        Nx, Ny = F.num, F.num

        Xn = np.linspace(-Nx / 2 * dx, Nx / 2 * dx, num=F.num) / mm
        Yn = np.linspace(-Ny / 2 * dy, Ny / 2 * dy, num=F.num) / mm

        ax2.plot(Xn, Phi1[int(F.num / 2)], "g", label="Phase")
        ax2.set_title("Phase")
        ax2.set_xlabel("mm")
        ax2.set_ylabel("phase, radians")

        ax1.plot(Xn, I1[int(F.num / 2)], "g", label="Int")
        ax1.set_title("Intensity")
        ax1.set_xlabel("mm")
        ax1.set_ylabel("Intensity rel.units")

        im3 = ax3.imshow(I1, extent=[Xn[0], Xn[-1], Yn[0], Yn[-1]])
        fig.colorbar(im3, ax=ax3)
        ax3.set_xlabel("mm")
        ax3.set_ylabel("mm")
        ax3.set_title("Intensity")

        im4 = ax4.imshow(Phi1, extent=[Xn[0], Xn[-1], Yn[0], Yn[-1]])
        fig.colorbar(im4, ax=ax4)
        ax4.set_title("Phase")
        ax4.set_xlabel("mm")
        ax4.set_ylabel("mm")

    else:
        ax2.plot(Phi1[int(F.num / 2)], "g", label="Phi")
        ax2.set_title("Phi")
        ax2.set_xlabel("pixels")
        ax2.set_ylabel("phase, radians")

        ax1.plot(I1[int(F.num / 2)], "g", label="Int")
        ax1.set_title("Intensity")
        ax1.set_xlabel("pixels")
        ax1.set_ylabel("Intensity rel.units")

        im3 = ax3.imshow(I1)
        fig.colorbar(im3, ax=ax3)
        ax3.set_title("Intensity")
        ax3.set_xlabel("pixels")
        ax3.set_ylabel("pixels")

        im4 = ax4.imshow(Phi1)
        fig.colorbar(im4, ax=ax4)
        ax4.set_title("Phase")
        ax4.set_xlabel("pixels")
        ax4.set_ylabel("pixels")

    if msg is None:
        if Fild.F_num is None:
            plt.figtext(0, 0.95, "Not propogated yet", fontsize=18, color="black")
        else:
            plt.figtext(
                0,
                0.95,
                "Propogated with F number = {Fild.F_num}",
                fontsize=18,
                color="black",
            )
    else:
        plt.figtext(0.05, 0.95, msg, fontsize=18, color="black")


def dispa(X):
    plt.imshow(np.abs(X))
    plt.show()


def dispph(X):
    plt.imshow(np.angle(X))
    plt.show()


def get_MTF(filename, lm, theta, xx, orient):
    """ Generates mirror tf from profile"""
    #     lm = 2.66e-9
    #     theta = 34.9e-3
    sinTheta = np.sin(theta)
    hdata = np.loadtxt(filename)
    x_s = hdata[:, 0]
    heights = hdata[:, 1]
    x_s_scaled = x_s * sinTheta
    x_s_scaled = x_s_scaled - (np.abs(x_s_scaled[-1]) - np.abs(x_s_scaled[0])) / 2
    opd = -4 * np.pi * (heights * sinTheta) / lm
    new_opd = np.interp(xx, x_s_scaled, opd, left=0, right=0)[0]
    MTF = np.zeros((len(new_opd), len(new_opd)), dtype=complex)
    MTF[:] = np.exp(1j * new_opd)
    if orient == "y":
        MTF = MTF.T
    return MTF
