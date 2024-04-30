import numpy as np
import scipy.constants as cst
import scipy.linalg as lin
import scipy.optimize as sci
from scipy.special import eval_genlaguerre as Ln
from scipy.special import eval_hermite as He
from scipy.special import factorial as fact
from scipy.special import kv as kv
from tqdm import tqdm
import matplotlib.pyplot as plt



class Fluxonium:

    '''
    This class compute various properties of the fluxonium qubit.
    Convention : Ec = e^2/2C // Ej = phiq**2/Lj // El = phiq**2/L
    and phiq = hbar/2e

    The hamiltonian is:

    H = 4Ecn^2 - Ejcos(phi + 2*pi*phi_ext) + El*phi**2/2

    !!! phi_ext = Phi_ext / (h/2e)

    '''


    def __init__(self, Ej, Ec, El, phi_ext, cutoff, Nj=None, wp = None, Ecc=None):


        self.Ej = Ej #Josephson Energy is GHz
        self.Ec = Ec #Charging Energy is GHz
        self.El = El #Inductive Energy is GHz
        self.phi_ext = phi_ext #External flux quanta
        self.cutoff = cutoff #Number of mode cutoff in the Fock basis


        if wp ==None:
            self.wp = 20 # plasma frequency of the junction array in GHz
        else:
            self.wp = wp

        if Nj == None:
            self.Nj = 100  # number of junction in the array
        else:
            self.Nj = Nj
        if Ecc ==None:
            self.Ecc = self.wp**2/(8*self.Nj*self.El) # charging eenergy of the array junctioins
        else:
            self.Ecc = Ecc



    def phase_op(self):

        '''
        Return the phase operator in the Fock basis

        '''

        Ec = self.Ec
        El = self.El
        cutoff = self.cutoff

        phi_0 = (8*Ec/El)**0.25

        sqrt_diag = np.sqrt(np.arange(1,cutoff))
        mat_op = np.diag(sqrt_diag, 1) + np.diag(sqrt_diag, -1)
        mat_op = mat_op*(phi_0/(np.sqrt(2)))

        return mat_op

    def charge_op(self):

        '''
        Return the charge operator in the Fock basis

        '''

        Ec = self.Ec
        El = self.El
        cutoff = self.cutoff

        phi_0 = (8*Ec/El)**0.25

        sqrt_diag = np.sqrt(np.arange(1,cutoff))
        mat_op = np.diag(sqrt_diag, -1) - np.diag(sqrt_diag, 1)
        mat_op = mat_op*(-1j/(np.sqrt(2)*phi_0))

        return mat_op



    def cos_a_phi_b(self, a, b):

        '''
        Return the cos(a*phi+b) operator where a and b are scalar

        '''

        Ej = self.Ej
        Ec = self.Ec
        El = self.El
        phi_ext = self.phi_ext
        cutoff = self.cutoff


        phase = self.phase_op()

        arg = a*phase + b*np.eye(cutoff)

        return lin.cosm(arg)


    def hamiltonian(self, Ej, Ec, El, phi_ext, cutoff):


        wr = np.sqrt(8*El*Ec) #Harmonic frequnecy
        phi_0 = (8*Ec/El)**0.25 # Phase fluctuation

        # Harmonic part of the hamiltonian
        er_vec = ((np.arange(cutoff, dtype='complex') ) + 0.5) * wr
        mat = np.diag(er_vec)

        cos_mat = self.cos_a_phi_b(a=1, b = 2*np.pi*phi_ext)


        mat-= Ej*(cos_mat)

        return mat



    def diago(self):

        '''
        Diagonalization of the fluxonium hamiltonian in the Fock basis

        Returns:    eigen_energy: Eigenvalues in GHz
                    eigen_vector: Eigenvector projected on the Fock basis
        '''

        Ej = self.Ej
        Ec = self.Ec
        El = self.El
        phi_ext = self.phi_ext
        cutoff = self.cutoff

        H_mat = self.hamiltonian(Ej, Ec, El, phi_ext, cutoff)

        eigen_energy, eigen_values = lin.eigh(H_mat)

        return eigen_energy, eigen_values
