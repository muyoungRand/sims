#%%
import numpy as np
import qutip as qt
import scipy.special as special

class operations():
    def __init__(self, nIons, nModes, nMax):
        self.nIons = nIons
        self.nModes = nModes
        self.nMax = nMax
        
        # Define Operators
        self.sM = []
        self.sZ = []
        for i in range(self.nIons):
            self.sM.append(self.def_sM_operator(i))
            self.sZ.append(self.def_sZ_operator(i))
            
        self.a = []    
        for i in range(self.nModes):
            self.a.append(self.def_a_operator(i))

    def eval_rabi_freq(self, nStart, nDelta, eta):
        """
        Calculates Rabi Frequency for nStart -> nStart + nDelta

        Args:
            nStart (int): Initial phonon state
            nDelta (int): Change in phonon number
            eta (float): Lamb-Dicke Parameter

        Returns:
            float: Rabi frequency
        """
        nEnd = nStart + nDelta

        nSmall = min([nStart, nEnd])
        nBig = max([nStart, nEnd])
        factor2 = np.exp(-0.5 * eta**2) * eta**(np.absolute(nDelta))
        factor3 = np.sqrt(np.math.factorial(nSmall)/np.math.factorial(nBig))
        factor4 = special.assoc_laguerre(eta**2, nSmall, np.absolute(nDelta))

        return factor2 * factor3 * factor4

    # Define Operators
    def def_sM_operator(self, index):
        sM = qt.qeye(1) # Don't know how to initialise empty object, so do this, then trace away
        
        # Define the sigmaMinus operator for the specific qubit labelled by 'index'
        for i in range(self.nIons):
            if i < index:
                sM = qt.tensor(sM, qt.qeye(2))
            elif i == index:
                sM = qt.tensor(sM, qt.destroy(2))
            elif i > index:
                sM = qt.tensor(sM, qt.qeye(2))
        
        # Tensor away initial 1-array        
        sM = qt.ptrace(sM, [1 + i for i in range(self.nIons)])
        
        # Add the motional parts
        for i in range(self.nModes):
            sM = qt.tensor(sM, qt.qeye(self.nMax))
            
        return sM

    def def_sZ_operator(self, index):
        sZ = qt.qeye(1) # Don't know how to initialise empty object, so do this, then trace away
        
        # Define the sigmaMinus operator for the specific qubit labelled by 'index'
        for i in range(self.nIons):
            if i < index:
                sZ = qt.tensor(sZ, qt.qeye(2))
            elif i == index:
                sZ = qt.tensor(sZ, qt.sigmaz())
            elif i > index:
                sZ = qt.tensor(sZ, qt.qeye(2))
                
        # Tensor away initial 1-array        
        sZ = qt.ptrace(sZ, [1 + i for i in range(self.nIons)])
        
        for i in range(self.nModes):
            sZ = qt.tensor(sZ, qt.qeye(self.nMax))
        return sZ
            
    def def_a_operator(self, index):
        a = qt.qeye(1)
        
        for i in range(self.nIons):
            a = qt.tensor(a, qt.qeye(2))
            
        a = qt.ptrace(a, [1 + i for i in range(self.nIons)])
        
        for i in range(self.nModes):
            if i < index:
                a = qt.tensor(a, qt.qeye(self.nMax))
            elif i == index:
                a = qt.tensor(a, qt.destroy(self.nMax))
            elif i > index:
                a = qt.tensor(a, qt.qeye(self.nMax))

        return a
    
    def def_carrier_hamiltonian(self, index):
        op = self.sM[index].dag()
        def op_coeff(t, args):
            return np.exp(1j * args['detuning_ion' + str(index)] * t) * np.exp(-1j * args['phase_ion' + str(index)])
        
        op_p = self.sM[index]
        def op_p_coeff(t, args):
            return np.exp(-1j * args['detuning_ion' + str(index)] * t) * np.exp(1j * args['phase_ion' + str(index)])
        
        carrier = [
            [op, op_coeff], [op_p, op_p_coeff]
        ]
        
        return carrier

    def def_rsb_hamiltonian(self, index_ion, index_mode):
        op = 1j * self.sM[index_ion].dag() * self.a[index_mode]
        def op_coeff(t, args):
            return np.exp(1j * (args['detuning_rsb_ion' + str(index_ion)] - args['freq_mode' + str(index_mode)]) * t) * np.exp(-1j * args['phase_ion' + str(index_ion)])
            
        op_p = -1j * self.sM[index_ion] * self.a[index_mode].dag()
        def op_p_coeff(t, args):
            return np.exp(-1j * (args['detuning_rsb_ion' + str(index_ion)] - args['freq_mode' + str(index_mode)]) * t) * np.exp(1j * args['phase_ion' + str(index_ion)])
        
        rsb = [
            [op, op_coeff], [op_p, op_p_coeff]
        ]
        
        return rsb
    
    def def_bsb_hamiltonian(self, index_ion, index_mode):
        op = 1j * self.sM[index_ion].dag() * self.a[index_mode].dag()
        def op_coeff(t, args):
            return np.exp(1j * (args['detuning_bsb_ion' + str(index_ion)] + args['freq_mode' + str(index_mode)]) * t) * np.exp(-1j * args['phase_ion' + str(index_ion)])
            
        op_p = -1j * self.sM[index_ion] * self.a[index_mode]
        def op_p_coeff(t, args):
            return np.exp(-1j * (args['detuning_bsb_ion' + str(index_ion)] + args['freq_mode' + str(index_mode)]) * t) * np.exp(1j * args['phase_ion' + str(index_ion)])
            
        bsb = [
            [op, op_coeff], [op_p, op_p_coeff]
        ]
        
        return bsb
# %%
