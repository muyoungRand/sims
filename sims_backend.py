#%%
import numpy as np
import qutip as qt
import scipy.special as special

class operations():
    def __init__(self, nIons, nModes, nMax, rabi, eta):
        self.nIons = nIons
        self.nModes = nModes
        self.nMax = nMax
        self.rabi = rabi
        self.eta = eta
        
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
        sZ = qt.qeye(1)
        
        for i in range(self.nIons):
            if i < index:
                sZ = qt.tensor(sZ, qt.qeye(2))
            elif i == index:
                sZ = qt.tensor(sZ, qt.sigmaz())
            elif i > index:
                sZ = qt.tensor(sZ, qt.qeye(2))
                        
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
        op = self.rabi * self.sM[index].dag()
        def op_coeff(t, args):
            return np.exp(-1j * args['detuning_ion' + str(index)] * t) * np.exp(1j * args['phase_ion' + str(index)])
        
        op_p = op.dag()
        def op_p_coeff(t, args):
            return np.conjugate(op_coeff(t, args))
        
        carrier = [
            [op, op_coeff], [op_p, op_p_coeff]
        ]
        
        return carrier

    def def_rsb_hamiltonian(self, index_ion, index_mode):
        op = (self.rabi / 2) * self.sM[index_ion].dag() * self.a[index_mode]
        def op_coeff(t, args):
            exp_freq = np.exp(-1j * t * (args['freq_mode' + str(index_mode)] + args['detuning_rsb_ion' + str(index_ion)]))
            exp_phase = np.exp(1j * args['phase_ion' + str(index_ion)])
            return exp_freq * exp_phase
            
        op_p = op.dag()
        def op_p_coeff(t, args):
            return np.conjugate(op_coeff(t, args))
        
        rsb = [
            [op, op_coeff], [op_p, op_p_coeff]
        ]
        
        return rsb
    
    def def_bsb_hamiltonian(self, index_ion, index_mode):
        op = (self.rabi / 2) * self.sM[index_ion].dag() * self.a[index_mode].dag()
        def op_coeff(t, args):
            exp_freq = np.exp(1j * t * (args['freq_mode' + str(index_mode)] - args['detuning_bsb_ion' + str(index_ion)]))
            exp_phase = np.exp(1j * args['phase_ion' + str(index_ion)])
            return exp_freq * exp_phase
            
        op_p = op.dag()
        def op_p_coeff(t, args):
            return np.conjugate(op_coeff(t, args))
            
        bsb = [
            [op, op_coeff], [op_p, op_p_coeff]
        ]
        
        return bsb
    
    def _sin2_pulse_(self, tPi, k):
        return 2 * np.sqrt(2) * np.sqrt( k * (1+k) * (2+k) / (1 + 3 * k * (2 + k)) ) * tPi
    
    def def_sin2_rsb_hamiltonian(self, index_ion, index_mode, tPi, k):
        op = (self.rabi / 2) * self.sM[index_ion].dag() * self.a[index_mode]

        dur = self._sin2_pulse_(tPi, k)

        def op_coeff(t, args):
            exp_freq = np.exp(-1j * t * (args['freq_mode' + str(index_mode)] + args['rsb_detuning']))
            exp_phase = np.exp(1j * args['phase_ion' + str(index_ion)])
            sin2 = np.sin(np.pi * t / dur)**2
            return sin2 * exp_freq * exp_phase
            
        op_p = op.dag()
        def op_p_coeff(t, args):
            return np.conjugate(op_coeff(t, args))
        
        sin2_rsb = [
            [op, op_coeff], [op_p, op_p_coeff]
        ]
        
        return sin2_rsb
    
    def def_sin2_bsb_hamiltonian(self, index_ion, index_mode, tPi, k):
        op = (self.rabi / 2) * self.sM[index_ion].dag() * self.a[index_mode].dag()

        dur = self._sin2_pulse_(tPi, k)

        def op_coeff(t, args):
            exp_freq = np.exp(1j * t * (args['freq_mode' + str(index_mode)] - args['bsb_detuning']))
            exp_phase = np.exp(1j * args['phase_ion' + str(index_ion)])
            sin2 = np.sin(np.pi * t / dur)**2
            return sin2 * exp_freq * exp_phase
            
        op_p = op.dag()
        def op_p_coeff(t, args):
            return np.conjugate(op_coeff(t, args))
            
        sin2_bsb = [
            [op, op_coeff], [op_p, op_p_coeff]
        ]
        
        return sin2_bsb
# ------------ Example - MS Gate ------------

# Initialise class
nIons = 2
nModes = 1
nMax = 100
rabi = 2 * np.pi * 0.5
eta = 0.1

op = operations(nIons, nModes, nMax, rabi, eta)

# System parameters
freq_mode = 0
detuning = 1.0
args = {
    'freq_mode0': 2 * np. pi * freq_mode,

    "detuning_ion0": 0.0,
    "detuning_rsb_ion0": -2 * np.pi * detuning,
    "detuning_bsb_ion0": 2 * np.pi * detuning,
    "phase_ion0": 0.0,
    
    "detuning_ion1": 0.0,
    "detuning_rsb_ion1": -2 * np.pi * detuning,
    "detuning_bsb_ion1": 2 * np.pi * detuning,
    'phase_ion1': 0.0
}

# Hamiltonians
carrier_ion0 = op.def_carrier_hamiltonian(index = 0)
bsb_ion0_mode0 = op.def_bsb_hamiltonian(index_ion = 0, index_mode = 0)
rsb_ion0_mode0 = op.def_rsb_hamiltonian(index_ion = 0, index_mode = 0)

carrier_ion1 = op.def_carrier_hamiltonian(index = 1)
bsb_ion1_mode0 = op.def_bsb_hamiltonian(index_ion = 1, index_mode = 0)
rsb_ion1_mode0 = op.def_rsb_hamiltonian(index_ion = 1, index_mode = 0)

H = bsb_ion0_mode0 + rsb_ion0_mode0 + bsb_ion1_mode0 + rsb_ion1_mode0