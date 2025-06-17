import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
factor = 1.5
plt.rcParams.update({
    "font.size": 18*factor,         # Default text size
    "axes.titlesize": 22*factor,    # Title font size
    "axes.labelsize": 20*factor,    # X/Y label font size
    "legend.fontsize": 16*factor,   # Legend font size
    "xtick.labelsize": 16*factor,   # X tick label size
    "ytick.labelsize": 16*factor    # Y tick label size
})
from scripts import spectral
from config import SEED_PARAMS
from config import OUTPUT_PARAMS
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
    if OUTPUT_PARAMS["norm"] == "backward":
        energy_real = np.sqrt(np.sum(np.abs(real_box[0])**2) + np.sum(np.abs(real_box[1])**2) + np.sum(np.abs(real_box[2])**2)) * dx**3
        energy_fourier = np.sqrt(np.sum(np.abs(fourier_box[0])**2) + np.sum(np.abs(fourier_box[1])**2) + np.sum(np.abs(fourier_box[2])**2)) * dx**3  /  ((nmax)**(3/2))
    elif OUTPUT_PARAMS["norm"] == "forward":
        energy_real = np.sqrt(np.sum(np.abs(real_box[0])**2) + np.sum(np.abs(real_box[1])**2) + np.sum(np.abs(real_box[2])**2)) * dx**3  /  ((nmax)**(3/2))
        energy_fourier = np.sqrt(np.sum(np.abs(fourier_box[0])**2) + np.sum(np.abs(fourier_box[1])**2) + np.sum(np.abs(fourier_box[2])**2)) * dx**3
    elif OUTPUT_PARAMS["norm"] == "ortho":
        energy_real = np.sqrt(np.sum(np.abs(real_box[0])**2) + np.sum(np.abs(real_box[1])**2) + np.sum(np.abs(real_box[2])**2)) * dx**3  /  ((nmax)**(3/4))
        energy_fourier = np.sqrt(np.sum(np.abs(fourier_box[0])**2) + np.sum(np.abs(fourier_box[1])**2) + np.sum(np.abs(fourier_box[2])**2)) * dx**3 /  ((nmax)**(3/4))
    
    print(f"Parseval's equality check for nmax={nmax}:")
    print(f"  Real space energy:    {energy_real:.6e}")
    print(f"  Fourier space energy: {energy_fourier:.6e}")
    print(f"  Ratio (should be ~1): {energy_real/energy_fourier:.6f}")

    # Power spectrum using our spectral function
    k_bins, P_k = spectral.power_spectrum_vector_field(real_box[0], real_box[1], real_box[2], dx=dx, ncores=OUTPUT_PARAMS["ncores"], norm=OUTPUT_PARAMS["norm"])
    k_bins = k_bins * 2 * np.pi
    integral_Pk = np.trapz(P_k, k_bins)
    print(f"Integral of the power spectrum for nmax={nmax}: {integral_Pk:.6e}")
    return k_bins, P_k, energy_real, energy_fourier, integral_Pk

def plot_power_spectrum(k_bins, P_k, nmax, energy_real, energy_fourier, integral_Pk, test_funtion, style=None):
    label = (
        f'$\\mathbf{{nmax}}$={nmax}\n'
        f'Real: {energy_real:.2e}\n'
        f'Fourier: {energy_fourier:.2e}\n'
        f'Ratio: {energy_real/energy_fourier:.3f}\n'
        f'∫P(k)dk: {integral_Pk:.2e}'
    )
    if style is None:
        style = {}
    plt.loglog(k_bins, P_k, label=label, **style)
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title("Power spectrum for different resolutions: " + test_funtion)
    plt.legend(fontsize=16, loc='best')

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
    
    styles = [
        {"color": "C3", "linestyle": "-", "alpha": 0.7},
        {"color": "C1", "linestyle": "-.", "alpha": 0.7},
        {"color": "C2", "linestyle": ":", "alpha": 0.7},
        {"color": "C0", "linestyle": "--", "alpha": 0.7},
    ]
    
    for idx, nmax in enumerate([64, 128, 256]):
        dx = SEED_PARAMS["size"] / nmax
        fourier_box = make_test_fourier_box(nmax, test_type=args.test_type)
        r_box = fft.ifftn(fourier_box[0], s=fourier_box[0].shape, norm=OUTPUT_PARAMS["norm"])
        r_boy = fft.ifftn(fourier_box[1], s=fourier_box[1].shape, norm=OUTPUT_PARAMS["norm"])
        r_boz = fft.ifftn(fourier_box[2], s=fourier_box[2].shape, norm=OUTPUT_PARAMS["norm"])
        real_box = [r_box, r_boy, r_boz]
        # Forward FFT to get back to Fourier space for comparison
        box_fft = fft.fftn(real_box[0], s=real_box[0].shape, norm=OUTPUT_PARAMS["norm"])
        boy_fft = fft.fftn(real_box[1], s=real_box[1].shape, norm=OUTPUT_PARAMS["norm"])
        boz_fft = fft.fftn(real_box[2], s=real_box[2].shape, norm=OUTPUT_PARAMS["norm"])
        fourier_box_fft = [box_fft, boy_fft, boz_fft]
        for i, comp in enumerate(['x', 'y', 'z']):
            fourier_box[i] = fourier_box[i].astype(np.complex64)
            fourier_box_fft[i] = fourier_box_fft[i].astype(np.complex64)
            diff = fourier_box[i] - fourier_box_fft[i]
            abs_diff = np.abs(diff)
            total_sum = np.sum(abs_diff)
            mean_diff = np.mean(abs_diff)
            std_diff = np.std(abs_diff)
            print(f"Component {comp}: sum(abs(diff)) = {total_sum:.3e}, mean(abs(diff)) = {mean_diff:.3e}, std(abs(diff)) = {std_diff:.3e}")
        # k_bins, P_k, energy_real, energy_fourier, integral_Pk = parceval_and_integral_checks(real_box, fourier_box, dx, nmax)
        # plot_power_spectrum(k_bins, P_k, nmax, energy_real, energy_fourier, integral_Pk, args.test_type, style=styles[idx % len(styles)])
        k_bins, P_k, energy_real, energy_fourier, integral_Pk = parceval_and_integral_checks(real_box, fourier_box_fft, dx, nmax)
        plot_power_spectrum(k_bins, P_k, nmax, energy_real, energy_fourier, integral_Pk, args.test_type, style=styles[idx % len(styles)])
    plt.show()