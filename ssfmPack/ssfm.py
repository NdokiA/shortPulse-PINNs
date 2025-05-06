import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq

class ssfm:
    def __init__(self):

        self.pulseError = 0 
        self.spectrumError = 0
        self.warning = True

    def getFreqRange(self, time: np.ndarray) -> np.ndarray:
        """
        Get frequency range from given time range

        Args:
            time (np.ndarray): time array

        Returns:
            np.ndarray: freq array
        """    
        d = time[1]-time[0]
        return fftshift(fftfreq(len(time), d = d))

    def getTimeRange(self, f: np.ndarray) -> np.ndarray:
        """
        Get time range from given frequency range

        Args:
            f (np.ndarray): freq array

        Returns:
            np.ndarray: time array
        """     
        d = f[1]-f[0]
        return fftshift(fftfreq(len(f), d = d))

    def getPower(self, pulse: np.ndarray) -> np.ndarray:
        """
        Get power of the pulse by taking the absolute square

        Args:
            pulse (np.ndarray): pulse array

        Returns:
            np.ndarray: power array
        """    
        return np.abs(pulse)**2

    def getEnergy(self, time: np.ndarray, pulse: np.ndarray) -> float:
        """
        Returns the energy of the pulse via trapezoid integration. 

        Args:
            time (np.ndarray): time array
            pulse (np.ndarray): pulse array

        Returns:
            float: energy of the pulse
        """    
        return np.trapz(self.getPower(pulse), time)

    def getSpectrum(self, time: np.ndarray, pulse: np.ndarray) -> np.ndarray:
        """
        Do Fourier Transformation to change the pulse (time domain) into spectrum (frequency domain)
        Asserts error if energy is not conserved up to a certain error during transformation

        Args:
            time (np.ndarray): time array
            pulse (np.ndarray): pulse array

        Returns:
            np.ndarray: spectrum array
        """    
        dt = time[1]-time[0]
        f = self.getFreqRange(time)
        spectrum = fftshift(fft(pulse))*dt

        pulseEnergy = self.getEnergy(time, pulse)
        spectrumEnergy = self.getEnergy(f, spectrum)
        error = np.abs(pulseEnergy/spectrumEnergy-1)
        self.spectrumError = max(self.spectrumError, error)
        
        if self.warning and self.spectrumError > 1e-6:
            print(f'Warning! Energy lost during fourier transformation reached threshold: {self.spectrumError:.3e}. This will only appear once')
            self.warning = False
            

        return spectrum

    def getPulse(self, f: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
        """
        Do Inverse Fourier Transformation to change the spectrum (freq. domain) into pulse (time domain)
        Asserts error if energy is not conserved up to a certain error during transformation

        Args:
            f (np.ndarray): frequency array
            spectrum (np.ndarray): spectrum array

        Returns:
            np.ndarray: pulse array
        """    
        time = self.getTimeRange(f)
        dt = time[1]-time[0] 
        pulse = ifft(ifftshift(spectrum))/dt 

        pulseEnergy = self.getEnergy(time, pulse)
        spectrumEnergy = self.getEnergy(f, spectrum)
        
        error = np.abs(pulseEnergy/spectrumEnergy-1)
        self.pulseError = max(self.pulseError, error)
        
        if self.warning and self.pulseError > 1e-6:
            print(f'Warning! Energy lost during fourier transformation reached threshold: {self.pulseError:.3e}. This will only appear once')
            self.warning = False

        return pulse
    
    def SSFM(self, t_array: np.ndarray, z_array: np.ndarray, init_pulse: np.ndarray, 
             alpha: float, betas: list, gamma: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform Split-Step Fourier Method (SSFM) to solve the Nonlinear Schrodinger Equation (NLSE)

        Args:
            t_array (np.ndarray): time array
            z_array (np.ndarray): spatial array
            init_pulse (np.ndarray): initial pulse
            alpha (float): nonlinear refractive index
            betas (list): fiber optic's dispersion coefficient
            gamma (float): fiber optic's attenuation coefficient

        Returns:
            tuple[np.ndarray, np.ndarray]: pulse matrix, spectrum matrix
        """    
        #Initialize pulse-spectrum matrix
        f = self.getFreqRange(t_array)
        dz = z_array[1]-z_array[0]

        z_dim = len(z_array)
        t_dim = len(t_array)
        pulseMatrix = np.zeros((z_dim, t_dim), dtype = complex)
        spectrumMatrix = np.zeros((z_dim+1, t_dim), dtype = complex)

        pulse = init_pulse.astype(complex)
        pulseMatrix[0] = pulse 
        spectrumMatrix[0] = self.getSpectrum(t_array, init_pulse)

        #Calculating SSFM
        omega = 2*np.pi*f
        dispersion = 1j*betas[0]/2*omega**2 -1j*betas[1]/6*omega**3
        dispersionLoss = np.exp((dispersion- alpha/2)*dz)
        nonlinearity = 1j*gamma*dz 

        for n in range(1,z_dim):
            pulse = pulse * np.exp(nonlinearity*self.getPower(pulse))
            spectrum = self.getSpectrum(t_array, pulse)*dispersionLoss 
            pulse = self.getPulse(f, spectrum)

            pulseMatrix[n,:] = pulse
            spectrumMatrix[n,:] = spectrum
        
        

        return pulseMatrix, spectrumMatrix