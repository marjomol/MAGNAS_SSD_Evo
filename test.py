import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
from scripts import spectral
from config import SEED_PARAMS
import argparse

def make_test_fourier_box(nmax, test_type="delta"):
    shape = (nmax, nmax, nmax)
    box = np.zeros(shape, dtype=np.complex64)
    boy = np.zeros(shape, dtype=np.complex64)
    boz = np.zeros(shape, dtype=np.complex64)
    if test_type == "delta":
        # Delta in Fourier space = constant in real space
        box[0, 0, 0] = 1.0
        boy[0, 0, 0] = 1.0
        boz[0, 0, 0] = 1.0
    elif test_type == "white_noise":
        # White noise in Fourier space = white noise in real space
        box = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
        boy = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
        boz = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    elif test_type == "constant":
        # Constant in Fourier space = delta function in real space
        box.fill(1.0)
        boy.fill(1.0)
        boz.fill(1.0)
    elif test_type == "sine_wave":
        # Sine wave in Fourier space = sine wave in real space
        kx = np.fft.fftfreq(nmax, d=1/nmax) * 2 * np.pi
        ky = np.fft.fftfreq(nmax, d=1/nmax) * 2 * np.pi
        kz = np.fft.fftfreq(nmax, d=1/nmax) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k2 = KX**2 + KY**2 + KZ**2
        box = np.sin(k2)
        boy = np.sin(k2)
        boz = np.sin(k2)
    elif test_type == "step_function":
        # Step function in Fourier space = sinc function in real space
        kx = np.fft.fftfreq(nmax, d=1/nmax) * 2 * np.pi
        ky = np.fft.fftfreq(nmax, d=1/nmax) * 2 * np.pi
        kz = np.fft.fftfreq(nmax, d=1/nmax) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        box = np.where(np.sqrt(KX**2 + KY**2 + KZ**2) < 1, 1.0, 0.0)
        boy = np.where(np.sqrt(KX**2 + KY**2 + KZ**2) < 1, 1.0, 0.0)
        boz = np.where(np.sqrt(KX**2 + KY**2 + KZ**2) < 1, 1.0, 0.0)
    elif test_type == "gaussian":
        # Gaussian in Fourier space = Gaussian in real space
        kx = np.fft.fftfreq(nmax, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(nmax, d=dx) * 2 * np.pi
        kz = np.fft.fftfreq(nmax, d=dx) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k2 = KX**2 + KY**2 + KZ**2
        sigma = nmax / 8
        box = np.exp(-k2 / (2 * sigma**2))
        boy = np.exp(-k2 / (2 * sigma**2))
        boz = np.exp(-k2 / (2 * sigma**2))
    elif test_type == "powerlaw":
        # Power-law spectrum
        kx = np.fft.fftfreq(nmax, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(nmax, d=dx) * 2 * np.pi
        kz = np.fft.fftfreq(nmax, d=dx) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k = np.sqrt(KX**2 + KY**2 + KZ**2)
        alpha = -2.9
        box = np.where(k > 0, k**(alpha/2), 0)
        boy = np.where(k > 0, k**(alpha/2), 0)
        boz = np.where(k > 0, k**(alpha/2), 0)
        
    BOX = [box, boy, boz]
    
    return BOX

def parceval_and_integral_checks(real_box, fourier_box, dx, nmax):
    
    # Parceval's theorem
    energy_real_x = np.sqrt(np.sum(np.abs(real_box[0])**2))
    energy_real_y = np.sqrt(np.sum(np.abs(real_box[1])**2))
    energy_real_z = np.sqrt(np.sum(np.abs(real_box[2])**2))
    energy_real = np.sqrt(energy_real_x**2 + energy_real_y**2 + energy_real_z**2)
    energy_fourier_x = np.sqrt(np.sum(np.abs(fourier_box[0])**2))
    energy_fourier_y = np.sqrt(np.sum(np.abs(fourier_box[1])**2))
    energy_fourier_z = np.sqrt(np.sum(np.abs(fourier_box[2])**2))
    energy_fourier = np.sqrt(energy_fourier_x**2 + energy_fourier_y**2 + energy_fourier_z**2)
    
    print(f"Parseval's equality check for nmax={nmax}:")
    print(f"  Real space energy:    {energy_real:.6e}")
    print(f"  Fourier space energy: {energy_fourier:.6e}")
    print(f"  Ratio (should be ~1): {energy_real/energy_fourier:.6f}")

    # Power spectrum using our spectral function
    k_bins, P_k = spectral.power_spectrum_vector_field(fourier_box[0], fourier_box[1], fourier_box[2], dx=dx)
    k_bins = k_bins * 2 * np.pi
    integral_Pk = np.trapz(P_k, k_bins)
    print(f"Integral of the power spectrum for nmax={nmax}: {integral_Pk:.6e}")
    return k_bins, P_k

def plot_power_spectrum(k_bins, P_k, nmax):
    plt.loglog(k_bins, P_k, label=f'nmax={nmax}')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.legend()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Fourier box types for power spectrum analysis.")
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["delta", "white_noise", "constant", "sine_wave", "step_function", "gaussian", "powerlaw"],
        default="gaussian",
        help="Type of test function to use in Fourier space."
    )
    args = parser.parse_args()
    
    for nmax in [64, 128, 256, 512]:
        dx = SEED_PARAMS["size"] / nmax
        fourier_box = make_test_fourier_box(nmax, test_type=args.test_type)
        r_box = fft.ifftn(fourier_box[0], s=fourier_box[0].shape, norm="ortho")
        r_boy = fft.ifftn(fourier_box[1], s=fourier_box[1].shape, norm="ortho")
        r_boz = fft.ifftn(fourier_box[2], s=fourier_box[2].shape, norm="ortho")
        real_box = [r_box, r_boy, r_boz]
        # Forward FFT to get back to Fourier space for power spectrum
        box_fft = fft.fftn(real_box[0], s=real_box[0].shape, norm="ortho")
        boy_fft = fft.fftn(real_box[1], s=real_box[1].shape, norm="ortho")
        boz_fft = fft.fftn(real_box[2], s=real_box[2].shape, norm="ortho")
        fourier_box_fft = [box_fft, boy_fft, boz_fft]
        k_bins, P_k = parceval_and_integral_checks(real_box, fourier_box, dx, nmax)
        plot_power_spectrum(k_bins, P_k, nmax)
    plt.title("Power spectrum for different resolutions")
    plt.show()