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


    def get_Ec(self):
        return self.Ec
    def get_El(self):
        return self.El
    def get_Ej(self):
        return self.Ej
    def get_cutoff(self):
        return self.cutoff
    def get_phiext(self):
        return self.phi_ext
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



    def sin_a_phi_b(self, a, b):

        '''
        Return the sin(a*phi+b) operator where a and b are scalar

        '''

        Ej = self.Ej
        Ec = self.Ec
        El = self.El
        phi_ext = self.phi_ext
        cutoff = self.cutoff


        phase = self.phase_op()

        arg = a*phase + b*np.eye(cutoff)

        return lin.sinm(arg)


    def exp_a_n(self, a):

        n = self.charge_op()
        arg = a*1j*n
        return lin.expm(arg)





    def hamiltonian(self, Ej, Ec, El, phi_ext, cutoff):


        wr = np.sqrt(8*El*Ec) #Harmonic frequnecy
        phi_0 = (8*Ec/El)**0.25 # Phase fluctuation

        # Harmonic part of the hamiltonian
        er_vec = ((np.arange(cutoff, dtype='complex') ) + 0.5) * wr
        mat = np.diag(er_vec)

        #phase operator
        # sqrt_diag = np.sqrt(np.arange(1,cutoff))
        # mat_sqrt = np.diag(sqrt_diag, 1) + np.diag(sqrt_diag, -1)
        # cosine part
        # phi_mat = 0.5*lin.expm(1j*phi_0/np.sqrt(2)*mat_sqrt)*np.exp(-1j*phi_ext*np.pi*2)
        # cos_mat = phi_mat + phi_mat.conj().T

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


    def fock_wavefunc_in_phase_space(self, i, phi_vec):

        '''
        Compute the wavefunction of Fock state i in the phase space


        Parameters:     i: wavefunction index
                        phi_vec: 1D-array of phase

        Return:         efn:1D-array for the ith wavefunction in phase space

        '''

        El = self.El
        Ec = self.Ec

        z = (El/(8*Ec))**0.5


        efn = 1/np.sqrt(fact(i)*2**i)*(z/(np.pi))**0.25* \
                np.exp(-z*phi_vec**2/(2))*He(i, (z)**0.5*phi_vec)

        return efn


    def wavefunction_in_phase(self, which, phi_vec):

        '''
        Compute the Fluxonium wavefunction in phase space

        Parameters:     which: 1D-list of the wavefunction to return
                        phi_vec: 1D-array phase vector

        Return:      psi_phi_space: 2D-array containing the wavefunctions

        '''

        Ej = self.Ej
        Ec = self.Ec
        El = self.El
        phi_ext = self.phi_ext
        cutoff = self.cutoff

        osc_wave = np.zeros((len(phi_vec),cutoff), dtype='complex')

        eigen_energy, eigen_values = self.diago()

        for n in tqdm(range(cutoff)):
            osc_wave[:, n] = self.fock_wavefunc_in_phase_space(n, phi_vec)

        psi_phi_space = np.real(np.dot(osc_wave, eigen_values))

        return psi_phi_space[:,which]


    def d_H_d_phi_ext(self):

        '''
        Return the derivative of the Hamiltonian with respect to
        an external flux

        '''


        Ej = self.Ej
        Ec = self.Ec
        El = self.El
        phi_ext = self.phi_ext

        phiq = cst.hbar/(2*cst.e)


        sin = self.sin_a_phi_b(a=1, b=2*np.pi*phi_ext)

        return -(Ej*cst.h*1e9)*(1/phiq)*sin
        # return -(Ej*cst.h*1e9)*sin*2*np.pi

    def d2_H_d2_phi_ext(self):

        '''
        Return the second derivative of the Hamiltonian with respect to
        an external flux

        '''

        Ej = self.Ej
        phi_ext = self.phi_ext

        phiq = cst.hbar/(2*cst.e)

        cos = self.cos_a_phi_b(a=1, b=2*np.pi*phi_ext)

        return (Ej*cst.h*1e9)*(1/phiq)**2*cos

    def d_H_d_Ic(self):

        '''
        Return the derivative of the Hamiltonian with respect to
        the critical current

        '''
        phi_ext = self.phi_ext


        phiq = cst.hbar/(2*cst.e)

        cos = self.cos_a_phi_b(a=1, b=2*np.pi*phi_ext)

        return -phiq*cos

    def matrix_element(self, operator, i='all', j='all'):

        '''
        Compute the matrix element of a given operator between to fluxonium eigenvector


        Parameters:     operator: str giving the operator name
                        i: index for the bra
                        j: index for the ket
                        the index are increasing with the energy of the corresponding
                        eigenvector

        return:

        '''

        Ej = self.Ej
        Ec = self.Ec
        El = self.El
        phi_ext = self.phi_ext
        cutoff = self.cutoff

        eigen_energy, eigen_values = self.diago()



        if operator=='charge':
            mat_op = self.charge_op()

        elif operator=='phase':
            mat_op = self.phase_op()

        elif operator=='cos_phi':
            mat_op = self.cos_a_phi_b(a=1, b=0)

        elif operator=='sin_phi':
            mat_op = self.sin_a_phi_b(a=1, b=0)

        else:
            mat_op = operator


        mat_ket = np.dot(mat_op, eigen_values)
        mat_element = np.dot(eigen_values.T, mat_ket)

        if i=='all' and j!='all':
            return mat_element[:,j]

        elif i!='all' and j=='all':
            return mat_element[i,:]

        elif j=='all'and i=='all':
            return mat_element

        else:
            return mat_element[i,j]

    def energies_and_elementsmatrix(self, operator, row_indices, column_indices) :
        eigen_energy, eigen_values = self.diago()
        Transistion_energies = np.zeros((len(row_indices), len(column_indices)))
        Elements_matrix = np.zeros((len(row_indices), len(column_indices)))
        if operator == '2e_charge' :
            mat_op = 2 *cst.e *  self.charge_op()
            mat_ket = np.dot(mat_op, eigen_values)
            Em = np.dot(eigen_values.T, mat_ket)
        elif operator =='phi0_phi' :
            mat_op = self.phase_op() * (cst.hbar/(2*cst.e))
            mat_ket = np.dot(mat_op, eigen_values)
            Em = np.dot(eigen_values.T, mat_ket)
        elif operator == '2phi0_sin_half_phi' :
            mat_op = self.sin_a_phi_b(a =0.5,b=0) * 2 * (cst.hbar/(2*cst.e))
            mat_ket = np.dot(mat_op, eigen_values)
            Em = np.dot(eigen_values.T, mat_ket)
        for i in row_indices :
            for j in column_indices:
                if i != j :
                    i, j = int(i) , int(j)
                    Transistion_energies[i,j] = eigen_energy[j]-eigen_energy[i]
                    Elements_matrix[i,j] = np.abs(Em[i,j])
                else :
                    i, j = int(i) , int(j)
                    Transistion_energies[i,j] = np.nan
                    Elements_matrix[i,j] = np.nan
        return Transistion_energies, Elements_matrix



    def relaxation_rate(self, source, i=1, j=0, T=100e-3, M=400*(cst.h/(2*cst.e)), xqp=3e-6,
                         Delta = 210e-6*cst.e, R_bias=50, R_charge=50):
        '''
        Compute the relaxation rate of a qubit

        Parameters:     i: index for the bra
                        j: index for the ket
                        source: (str) noise source
                        T: Circuit temperature
                        M: Mutual inductance in USI
                        xqp: quasiparticle density
                        Delta: gap of the superconductor
                        R_bias: impedance of the line bias (can be complex)
                        R_charge: impedance of the coupled impedance (can be complex)


        return:
                Relaxation rate in Hz
        '''

        Ej = self.Ej*1e9*cst.h
        Ec = self.Ec*1e9*cst.h
        El = self.El*1e9*cst.h
        phi_ext = self.phi_ext

        if T ==0:
            T=1e-4

        eigen_energy, eigen_values = self.diago()


        w_ij = (eigen_energy[i] - eigen_energy[j])*2*np.pi*1e9


        if source == 'capacitive':

            # --- Spectrum

            Q_cap = 1e6*(2*np.pi*6e9/(np.abs(w_ij)))**0.7

            Spec = 16*cst.hbar*Ec/Q_cap*(w_ij/np.abs(w_ij))
            # Admittance =
            # Spec = cst.hbar*w_ij * Admittance* (np.coth((cst.hbar*w_ij)/(2*cst.k*T))+1)

            # the w_ij/abs(wij) is to reserve Spec symetry

            # --- Operator

            noise_op = 'charge'

        elif source == 'inductive':

            # --- Spectrum


            x = cst.h*0.5e9/(2*cst.k*T)
            y = cst.hbar*np.abs(w_ij)/(2*cst.k*T)

            Q_ind = 500e6*(kv(0,x)*np.sinh(x))/(kv(0,y)*np.sinh(y))
            Spec = 2*(El*cst.hbar)/Q_ind *(w_ij/np.abs(w_ij))

            # the w_ij/abs(wij) is to reserve Spec symetry

            # --- Operator

            noise_op = 'phase'

        elif source == 'charge_impedance':

            # print('not done yet')
            Spec = cst.hbar*w_ij*(1+ np.coth(cst.hbar*np.abs(w_ij)/(2*cst.k*T)))/np.real(R_charge)
            noise_op = 'charge'

        elif source =='flux_bias_line':


            # --- Spectrum

            # Spec = M**2*cst.hbar*w_ij / R_bias
            Spec = 2*cst.hbar*w_ij*M**2/np.real(R_bias)
            # print(Spec)

            # --- Operator
            # phiq = cst.hbar/(2*cst.e)

            # noise_op = self.sin_a_phi_b(a=1, b=2*np.pi*phi_ext)
            # # noise_op *= -Ej/phiq
            # noise_op *= (-Ej/phiq)
            noise_op = self.d_H_d_phi_ext()

        elif source =='quasiparticle':

            print('not working for know')

# --------------------------------------------------------------------------------------
            # # --- Spectrum

            # # This formula is giving is weird temperature dependance, 1/Y increase with T
            # Rq = cst.h/(cst.e**2)
            # # Re_Yqp_old = np.sqrt(2/np.pi)*(8*Ej)/(Rq*Delta)*(2*Delta/(cst.hbar*np.abs(w_ij)))**3/2 *xqp*np.sqrt(beta)*kv(0, beta)*np.sinh(beta)
            # Re_Yqp_old = np.sqrt(2/np.pi)*(8*Ej)/(Rq*Delta)*(2*Delta/(cst.hbar*np.abs(w_ij)))**3/2*np.sqrt(np.pi)*np.sqrt(cst.k*T/(cst.hbar*np.abs(w_ij)))*xqp

            # # # Taken from Cattelani et al (2011), used in Pop et al (2014)
            # # Lj = (cst.hbar/(2*cst.e))**2/Ej
            # # Re_Yqp = xqp/(np.abs(w_ij)*Lj*np.pi)*np.sqrt((2*Delta)/(cst.hbar*np.abs(w_ij)))
            # Spec = 2*cst.hbar*w_ij*Re_Yqp_old



            # def y_qp_fun(omega):
            #     """
            #     Based on Eq. S23 in the appendix of Smith et al (2020).
            #     """
            #     # Note that y_qp_fun is always symmetric in omega, i.e. In Smith et al 2020,
            #     # we essentially have something proportional to sinh(omega)/omega
            #     omega = abs(omega)

            #     therm_ratio = cst.hbar*omega / (cst.k*T)

            #     re_y_qp = (
            #         np.sqrt(2 / np.pi)
            #         * (8 / Rq)
            #         * (Ej/ Delta)
            #         * (2 * Delta / (cst.hbar*omega)) ** (3 / 2)
            #         * xqp
            #         * np.sqrt(1 / 2 * therm_ratio)
            #         * kv(0, 1 / 2 * abs(therm_ratio))
            #         * np.sinh(1 / 2 * therm_ratio)
            #     )

            #     return re_y_qp



            # def spectral_density(omega):
            #     """Based on Eq. 19 in Smith et al (2020)."""
            #     therm_ratio = cst.hbar*omega / (cst.k*T)

            #     return (
            #         2
            #         * omega *cst.hbar
            #         * complex(y_qp_fun(omega)).real
            #         * (1 / np.tanh(0.5 * therm_ratio))
            #         / (1 + np.exp(-therm_ratio))
            #     )

            # Spec = spectral_density(w_ij)

# ---------------------------------------------------------------------------------

            # def y_qp_func(omega):

            #     omega = np.abs(omega)
            #     beta = (cst.hbar*omega)/(2*cst.k*T)
            #     Rq = cst.h/(cst.e**2)

            #     re_y_qp = (
            #         np.sqrt(2 / np.pi)
            #         * (8 / Rq)
            #         * (Ej / Delta)
            #         * (2 * Delta / (cst.hbar*omega)) ** (3 / 2)
            #         * xqp
            #         * np.sqrt(1 / 2 * beta)
            #         * kv(0, 1 / 2 * np.abs(beta))
            #         * np.sinh(1 / 2 * beta))

            #     return re_y_qp



            # Spec = 2*cst.hbar*w_ij*complex(y_qp_func(w_ij)).real
            # Spec = 2*cst.hbar*w_ij*Re_Yqp


            # --- Operator


            phiq = cst.hbar / (2*cst.e)
            noise_op = self.sin_a_phi_b(a=0.5, b=np.pi*phi_ext)
            noise_op *= 2*phiq

        else:
            print('Relaxation source not known')

        # Spec= Spec/(1-np.exp(-cst.hbar*w_ij/(cst.k*T)))

        mat_ij = self.matrix_element(noise_op, i, j)

        return 1/cst.hbar**2*np.abs(mat_ij)**2*Spec
        # return np.abs(mat_ij)**2


    def T1(self, source='all', i=1, j=0, T=100e-3, M=400*(cst.h/(2*cst.e)), xqp=3e-6,
                         Delta = 210e-6*cst.e, R_bias=50, R_charge=50):

        '''
        Compute the T1 of a qubit

        Parameters:     i: index for the bra
                        j: index for the ket
                        source: (str) noise source
                        T: Circuit temperature
                        M: Mutual inductance in USI
                        xqp: quasiparticle density
                        Delta: gap of the superconductor
                        R_bias: impedance of the line bias (can be complex)
                        R_charge: impedance of the coupled impedance (can be complex)


        return: T1 in second

        note: I'm not sure about the exactness of 'flux_bias_noise'

        '''

        kwargs = {'T':T, 'M':M, 'xqp':xqp, 'Delta':Delta, 'R_bias':R_bias, 'R_charge':R_charge}

        rate = 0

        if source=='all':

            # source = ['capacitive',
            #           'inductive',
            #           'flux_bias_line',
            #           'quasiparticle']

            source = ['capacitive',
                      'inductive',
                      'flux_bias_line','charge_impedance']


        for s in source:

                rate_ij = self.relaxation_rate(s, i, j, **kwargs)
                rate_ji = self.relaxation_rate(s, j, i, **kwargs)

                rate += rate_ij + rate_ji

        return 1/rate


    def dephasing_rate(self, source, i=1, j=0, A_flux=1e-6*cst.h/(2*cst.e), A_Ic =1e-7, w_IR=1*2*np.pi, w_UV=2*np.pi*3e9, time=10e-6):

        """
        Facteur Ej par rapport à scqubit

        """


        if source=='flux':

            noise_op_1 = self.d_H_d_phi_ext()
            noise_op_2 = self.d2_H_d2_phi_ext()

            dE0_d_phi_ext = self.matrix_element(noise_op_1, j, j)
            dE1_d_phi_ext = self.matrix_element(noise_op_1, i, i)

            d2E0_d2_phi_ext = self.matrix_element(noise_op_2, j, j)
            d2E1_d2_phi_ext = self.matrix_element(noise_op_2, i, i)

            d_wij_d_phi_exp = np.abs(dE0_d_phi_ext - dE1_d_phi_ext)/cst.hbar
            d2_wij_d2_phi_exp = np.abs(d2E0_d2_phi_ext - d2E1_d2_phi_ext)/cst.hbar

            r_1 = 2*A_flux**2*d_wij_d_phi_exp**2*np.abs(np.log(w_IR*time))
            r_2 = 2*A_flux**4*d2_wij_d2_phi_exp**2*(np.log(w_UV/w_IR)**2 + 2*np.abs(np.log(w_IR*time))**2)


            T_phase = (r_1 + r_2)**(-0.5)
            rate_phase = 1/T_phase



        elif source=='critical_current':

            #print(A_Ic)

            Ej = self.Ej*cst.h*1e9
            Ic = Ej/(cst.hbar/(2*cst.e))
            A_Ic *=Ic

            noise_op = self.d_H_d_Ic()

            dE0_d_Ic = self.matrix_element(noise_op, i, i)
            dE1_d_Ic = self.matrix_element(noise_op, j, j)

            d_wij_d_Ic = np.abs(dE0_d_Ic - dE1_d_Ic)/cst.hbar

            # print(A_Ic*d_wij_d_Ic, A_Ic, d_wij_d_Ic)

            rate_phase = A_Ic*d_wij_d_Ic*np.sqrt(np.abs(2*np.log(w_IR*time)))



        elif source=='phase_slip':

            Nj = self.Nj
            Ejj = self.El*cst.h*1e9*Nj
            Ecc = self.Ecc*cst.h*1e9
            wp = self.wp*1e9*2*np.pi

            z = np.sqrt(2*Ecc/Ejj)/np.pi
            # print(z)
            de_0 =8*np.sqrt(2)*cst.hbar*wp*np.exp(-4/(np.pi*z))/np.sqrt(np.pi*z)
            # print(de_0/cst.h*1e-9)


            # if (de_0/cst.h*1e-9*Nj>self.El) or z>0.15:
            #     print('Dephasing is overestimated')
            #     print('E_phase_slip/El = %.1f'%(de_0/cst.h*1e-9*Nj/self.El))
            #     print('z = %.1f'%z)

            T = self.exp_a_n(a=-2*np.pi)

            T_i = self.matrix_element(T, i, i)
            T_j = self.matrix_element(T, j, j)

            dw_ij = 2*Nj*de_0*np.abs(T_j - T_i)/cst.hbar
            # print(dw_ij*1e-9/(2*np.pi))
            # print(np.abs(T_j - T_i))

            rate_phase = dw_ij / (4*np.sqrt(Nj))




        else:

            print('Dephasing source not known')

        return rate_phase


    def T_phase(self, source, i=1, j=0, A_flux=1e-6*cst.h/(2*cst.e), A_Ic =1e-7, w_IR=1*2*np.pi, w_UV=2*np.pi*3e9, time=10e-6):



        kwargs = {'i':i, 'j':j, 'A_flux': A_flux, 'A_Ic':A_Ic, 'w_IR':w_IR, 'w_UV':w_UV, 'time':time}

        if source =='flux':

            rate_phase = self.dephasing_rate(source, **kwargs)

        elif source =='critical_current':

            rate_phase = self.dephasing_rate(source, **kwargs)

        elif source=='phase_slip':
            rate_phase = self.dephasing_rate(source, **kwargs)

        elif source=='all':

            rate_phase = 0
            source = ['flux', 'critical_current']

            for s in source:
                rate_phase +=self.dephasing_rate(s, **kwargs)

        else:

            print('Dephasing source not known')


        return 1/rate_phase
