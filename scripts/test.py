"""
MAGNAS SSD Evolution
A tool to analyse simulated cosmological magnetic field induction and the Small Scale Dynamo amplification.

test module
Defines test magnetic and velocity fields from which to derive the analytical expressions of the components of the induction equation
in terms of the magnetic energy to check for errors and infer how well would it work with unknown simulated fields.

Created by Marco Molina Pradillo
"""
import numpy as np
import scripts.readers as reader

def test_limits(a0, dir_grids, dir_params, sims, IT, nmax, size):
    """
    This function defines some constant parameters and the space for the test fields.
    
    Args:
        - a0: Scale factor at z=0.
        - dir_grids: Directory where the grid files are located.
        - dir_params: Directory where the parameters file is located.
        - sims: Name of the simulation.
        - IT: List of iteration numbers to read. All iterations must be provided to calculate the limits of t and z.
        - nmax: Maximum number of cells in one dimension of the grid.
        - size: Physical size of the grid in comoving units (e.g., Mpc/h).

    Returns:
        - x_test, y_test, z_test: 3D grid coordinates.
        - k: Wave number for the sinusoidal test fields.
        - ω: Angular frequency for the sinusoidal test fields.
        
    Author: Marco Molina Pradillo
    """
    # Create a 3D commoving grid
    x_test, y_test, z_test = np.mgrid[-size/2:size/2:(nmax+1)*1j, -size/2:size/2:(nmax+1)*1j, -size/2:size/2:(nmax+1)*1j]
    x_test = (x_test[:-1, :-1, :-1] + x_test[1:, :-1, :-1]) / 2
    y_test = (y_test[:-1, :-1, :-1] + y_test[:-1, 1:, :-1]) / 2
    z_test = (z_test[:-1, :-1, :-1] + z_test[:-1, :-1, 1:]) / 2
    
    grid_zeta = []
    grid_time = []
    a = []
    
    # Read grid data using the reader
    for it in IT:
        grid = reader.read_grids(
            it=it,
            path=dir_grids + sims,
            parameters_path=dir_params,
            digits=5,
            read_general=True,
            read_patchnum=False,
            read_dmpartnum=False,
            read_patchcellextension=False,
            read_patchcellposition=False,
            read_patchposition=False,
            read_patchparent=False,
            nparray=True
        )
        # Unpack grid data with explicit variable names for clarity
        (
            _,
            t,
            _,
            _,
            z,
            *_
        ) = grid
        
        a_it = a0 / (1 + z)  # Scale factor at the redshift zeta
        
        grid_zeta.append(z)
        grid_time.append(t)
        a.append(a_it)
    
    # Calculate minimum and maximum of t and z
    t_min = np.min(grid_time)
    t_max = np.max(grid_time)
    t_min_index = np.argmin(grid_time)
    t_max_index = np.argmax(grid_time)
    z_min = np.min(grid_zeta)
    z_max = np.max(grid_zeta)
    
    # Constants for the range of kz + ωt
    min_angle = np.pi / 2 # For min_angle: k*z_min*a(t_min) + ω*t_min = min_angle
    max_angle = np.pi - 0.01  # Slightly less than π # For max_angle: k*z_max*a(t_max) + ω*t_max = max_angle
    
    # Solve for ω and k based on the desired angle range
    k = (max_angle - min_angle * (t_max)/(t_min)) / (z_max * a[t_max_index] - z_min * a[t_min_index] * (t_max)/(t_min)) 
    ω = (min_angle - k * z_min * a[t_min_index]) / (t_min)
    
    # Test simulation parameters
    rx = - size / 2 + size / nmax
    
    clus_cr0amr_test = np.ones_like(x_test)
    clus_solapst_test = np.ones_like(x_test)
    grid_patchrx_test = np.array([rx])
    grid_patchry_test = np.array([rx])
    grid_patchrz_tets = np.array([rx])
    grid_patchnx_test = np.array([nmax])
    grid_patchny_test = np.array([nmax])
    grid_patchnz_test = np.array([nmax])
    grid_npatch_test = np.array([0])

    return x_test, y_test, z_test, k, ω, clus_cr0amr_test, clus_solapst_test, grid_patchrx_test, grid_patchry_test, grid_patchrz_tets, grid_patchnx_test, grid_patchny_test, grid_patchnz_test, grid_npatch_test


def numeric_test_fields(grid_time, grid_npatch, a, H, test_params):
    """
    Defines simple sinusoidal test magnetic and velocity fields of the form:
        Magnetic field:
            $$  \tilde{\vec{B}}(x,y,z,t) = \begin{pmatrix} \frac{B_0\sin(kza + \omega t)}{\sqrt{ \rho_{B} }} \\ 0 \\ 0 \end{pmatrix} $$
        Velocity field:
            $$  \vec{v}(x,y,z,t) = \begin{pmatrix} 0 \\ -\omega ya \cot(kza+\omega t) \\ 0 \end{pmatrix} - \dot{a}\vec{x}$$

    Args:
        - grid_time: List of times for each iteration.
        - grid_npatch: List of number of patches for each iteration.
        - a: Scale factor at the current iteration.
        - H: Hubble parameter at the current iteration.
        - test_params: Dictionary containing the parameters for the test fields:
            - x_test, y_test, z_test: 3D grid coordinates.
            - k: Wave number for the sinusoidal test fields.
            - ω: Angular frequency for the sinusoidal test fields.
            - B0: Amplitude of the magnetic field.

    Returns:
        - bx, by, bz: Components of the magnetic field.
        - vx, vy, vz: Components of the velocity field.

    Author: Marco Molina Pradillo
    """
    x = test_params['x_test']
    y = test_params['y_test']
    z = test_params['z_test']
    k = test_params['k']
    ω = test_params['ω']
    B0 = test_params['B0']

    # Magnetic field components
    bx = [B0 * np.sin(k * z * a + ω * grid_time) for _ in range(1 + sum(grid_npatch))]
    by = np.zeros_like(bx)
    bz = np.zeros_like(bx)

    # Velocity field components
    vx = [- H * a * x for _ in range(sum(grid_npatch)+1)]
    vy = [- ω * y * a * (1/np.tan(k * z * a + ω * grid_time)) - H * a * y for _ in range(sum(grid_npatch)+1)]
    vz = [- H * a * z for _ in range(sum(grid_npatch)+1)]

    return bx, by, bz, vx, vy, vz

def analytic_test_fields(grid_time, grid_npatch, a, H, Bx, test_params):
    """
    Defines the analytical expressions of the components of the induction energy equation for the test fields defined in numeric_test_fields.
    The components are defined as follows:
        Divergence:
            $$	\frac{1}{a} \tilde{\vec{B}} \cdot \vec{v} ( \vec{\nabla} \cdot \tilde{\vec{B}} ) = 0$$
        Compression:
            $$	\frac{1}{a} \frac{{B_0}^{2} \sin^{2}(kza + \omega t)}{\rho_{B}} (\omega a \cot(kza + \omega t) + 3 \dot{a}) $$
        Stretching:
            $$	\frac{1}{a} \tilde{\vec{B}} \cdot (\tilde{\vec{B}} \cdot \vec{\nabla}) \vec{v} = -H \frac{B_0^2\sin^2(kza + \omega t)}{\rho_{B}}$$
        Advection:
            $$	-\frac{1}{a} \tilde{\vec{B}} \cdot (\vec{v} \cdot \vec{\nabla}) \tilde{\vec{B}} = \dot{a} \frac{{B_0}^{2} k}{\rho_{B}} \sin(kza + \omega t) \cos(kza + \omega t)$$
        Cosmic Drag:
            $$	-\frac{H}{2} \tilde{\vec{B}}^2 = -\frac{H}{2} \frac{B_0^2\sin^2(kza + \omega t)}{\rho_{B}}$$
        Total Compacted Induction:
            $$	\frac{\tilde{\vec{ B }}}{a} \vec{\nabla} \times ( \vec{v} \times \tilde{\vec{ B }} ) - \frac{H}{2} {\tilde{\vec{ B }}}^{2} = \frac{{B_0}^{2}sin^{2}(kza + \omega t)}{{\rho_{B}}} \left[(\omega + k\dot{a}z) \cot(kza+\omega t) + \frac{3}{2}H \right]	$$

    Args:
        - grid_time: List of times for each iteration.
        - grid_npatch: List of number of patches for each iteration.
        - a: Scale factor at the current iteration.
        - H: Hubble parameter at the current iteration.
        - Bx: x-component of the magnetic field from numeric_test_fields.
        - test_params: Dictionary containing the parameters for the test fields:
            - test: boolean to use test fields or not
            - x_test, y_test, z_test: 3D grid coordinates.
            - k: Wave number for the sinusoidal test fields.
            - ω: Angular frequency for the sinusoidal test fields.
            - B0: Amplitude of the magnetic field.

    Returns:
        compression, divergence, stretching, advection, cosmic_drag, total
    """

    x = test_params['x_test']
    y = test_params['y_test']
    z = test_params['z_test']
    k = test_params['k']
    ω = test_params['ω']
    
    results = {}
    
    # Compression
    results['MIE_compres_B2'] = [(1/a) * (Bx[p])**2 * (ω * a * (1/np.tan(k * z * a + ω * grid_time)) + 3 * H * a) for p in range(sum(grid_npatch)+1)]
    # Divergence is given to be 0, no calculation needed
    results['MIE_diver_B2'] = np.zeros_like(results['MIE_compres_B2'])
    # Stretching
    results['MIE_stretch_B2'] = [- H * Bx[p]**2 for p in range(sum(grid_npatch)+1)]
    # Advection
    results['MIE_advec_B2'] = [H * a * (Bx[p]**2 /np.sin(k * z * a + ω * grid_time)) * k * np.cos(k * z * a + ω * grid_time) for p in range(sum(grid_npatch)+1)]
    # Cosmic Drag
    results['MIE_drag_B2'] = [-(H/2) * Bx[p]**2 for p in range(sum(grid_npatch)+1)]
    # Total Compact Induction
    results['MIE_total_B2'] = [Bx[p]**2 * ((ω + k * H * a * z) * (1/np.tan(k * z * a + ω * grid_time)) + ((3*H)/2)) for p in range(sum(grid_npatch)+1)]
    
    results['kinetic_energy_density'] = [0.5 * ( (H * a * z)**2 + (ω * y * a * (1/np.tan(k * z * a + ω * grid_time)) + H * a * y)**2 + (H * a * z)**2 ) for _ in range(sum(grid_npatch)+1)]
    results['int_b2'] = [np.mean(Bx[p]**2) for p in range(sum(grid_npatch)+1)]
    
    return results