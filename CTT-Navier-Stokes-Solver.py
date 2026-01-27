"""
CTT Navier-Stokes Spectral Solver
Implements Convergent Time Theory (CTT) formulation of 3D incompressible NS
with fractal temporal layers (L=33) and dispersion coefficient α=0.0302011
"""

import numpy as np
import time

class CTT_NavierStokes_Solver:
    """
    Spectral solver for CTT formulation of 3D incompressible Navier-Stokes
    Maps classical time t → temporal layers d with energy decay E(d) = E₀e^{-αd}
    """
    
    def __init__(self, res=32, alpha=0.0302011, layers=33, nu=0.001):
        """
        Initialize CTT-NS solver
        
        Parameters:
        -----------
        res : int
            Spatial resolution (res³ grid)
        alpha : float
            CTT temporal dispersion coefficient
        layers : int
            Number of temporal layers (default 33)
        nu : float
            Kinematic viscosity in CTT mapping
        """
        self.res = res
        self.alpha = alpha
        self.L = layers
        self.nu = nu
        
        # CTT constants
        self.decay_factors = np.exp(-alpha * np.arange(layers))
        
        # Spectral grid
        k = np.fft.fftfreq(res, 1/(2*np.pi))
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        self.k_sq = self.kx**2 + self.ky**2 + self.kz**2
        self.k_sq[0,0,0] = 1.0  # Avoid division by zero
        
        # Initialize vorticity field (solenoidal in Fourier space)
        self.init_vorticity()
        
    def init_vorticity(self):
        """Initialize divergence-free vorticity field in Fourier space"""
        # Random initial vorticity
        w_hat = np.random.randn(self.res, self.res, self.res, 3) + \
                1j * np.random.randn(self.res, self.res, self.res, 3)
        
        # Project to divergence-free (∇·ω = 0)
        k_dot_w = (self.kx * w_hat[...,0] + 
                   self.ky * w_hat[...,1] + 
                   self.kz * w_hat[...,2])
        
        w_hat[...,0] -= (self.kx / self.k_sq) * k_dot_w
        w_hat[...,1] -= (self.ky / self.k_sq) * k_dot_w
        w_hat[...,2] -= (self.kz / self.k_sq) * k_dot_w
        
        self.w_hat = w_hat
        
    def compute_nonlinear_term(self, w_phys):
        """
        Compute (ω·∇)ω term using pseudo-spectral method
        Returns Fourier transform of nonlinear term
        """
        # Gradient in physical space
        grad_w = np.gradient(w_phys, axis=(0,1,2))
        
        # ω·∇ω in physical space
        nonlinear = np.zeros_like(w_phys)
        for i in range(3):
            for j in range(3):
                nonlinear[...,i] += w_phys[...,j] * grad_w[j][...,i]
        
        # Transform to Fourier space
        nonlinear_hat = np.fft.fftn(nonlinear, axes=(0,1,2))
        
        # Project to divergence-free
        k_dot_nl = (self.kx * nonlinear_hat[...,0] + 
                    self.ky * nonlinear_hat[...,1] + 
                    self.kz * nonlinear_hat[...,2])
        
        nonlinear_hat[...,0] -= (self.kx / self.k_sq) * k_dot_nl
        nonlinear_hat[...,1] -= (self.ky / self.k_sq) * k_dot_nl
        nonlinear_hat[...,2] -= (self.kz / self.k_sq) * k_dot_nl
        
        return nonlinear_hat
    
    def step_layer(self, d):
        """
        Advance one temporal layer using CTT equations:
        ∂ω/∂d + α(ω·∇)ω = α∇²ω
        """
        # Current vorticity in physical space
        w_phys = np.fft.ifftn(self.w_hat, axes=(0,1,2)).real
        
        # Compute nonlinear term
        nl_hat = self.compute_nonlinear_term(w_phys)
        
        # CTT update rule
        dt = 0.01  # Layer step size
        
        # Right-hand side: -α(ω·∇)ω + α∇²ω
        rhs_hat = -self.alpha * nl_hat - self.alpha * self.k_sq[...,None] * self.w_hat
        
        # Update vorticity
        self.w_hat += dt * rhs_hat
        
        # Apply CTT energy decay
        self.w_hat *= self.decay_factors[d]
        
        # Enforce solenoidality
        k_dot_w = (self.kx * self.w_hat[...,0] + 
                   self.ky * self.w_hat[...,1] + 
                   self.kz * self.w_hat[...,2])
        
        self.w_hat[...,0] -= (self.kx / self.k_sq) * k_dot_w
        self.w_hat[...,1] -= (self.ky / self.k_sq) * k_dot_w
        self.w_hat[...,2] -= (self.kz / self.k_sq) * k_dot_w
        
    def solve(self, steps_per_layer=10):
        """
        Solve CTT-NS equations across all temporal layers
        
        Returns:
        --------
        max_vorticity : list
            Maximum vorticity magnitude at each layer
        energy : list
            Total energy at each layer
        """
        max_vort = []
        energy = []
        
        print(f"Solving CTT Navier-Stokes (L={self.L}, α={self.alpha}, {self.res}³ grid)")
        print("=" * 60)
        
        for d in range(self.L):
            layer_start = time.time()
            
            for _ in range(steps_per_layer):
                self.step_layer(d)
            
            # Compute diagnostics
            w_phys = np.fft.ifftn(self.w_hat, axes=(0,1,2)).real
            vort_mag = np.sqrt(np.sum(w_phys**2, axis=-1))
            max_vort.append(np.max(vort_mag))
            
            # CTT energy E(d) = 1/2 ∫|ω|² dx
            E = 0.5 * np.sum(w_phys**2) / self.res**3
            energy.append(E)
            
            layer_time = time.time() - layer_start
            print(f"Layer {d+1:2d}/{self.L}: E = {E:.6f}, |ω|∞ = {max_vort[-1]:.6f}, time = {layer_time:.3f}s")
        
        print("=" * 60)
        print(f"Final energy decay ratio: E({self.L})/E(0) = {energy[-1]/energy[0]:.6f}")
        print(f"Predicted CTT decay: exp(-αL) = {np.exp(-self.alpha*self.L):.6f}")
        
        return max_vort, energy

# Example usage
if __name__ == "__main__":
    # Create solver
    solver = CTT_NavierStokes_Solver(res=32, alpha=0.0302011, layers=33)
    
    # Run simulation
    max_vort, energy = solver.solve(steps_per_layer=5)
    
    # Verify CTT energy bound
    E0 = energy[0]
    predicted = [E0 * np.exp(-solver.alpha * d) for d in range(solver.L)]
    
    print("\nCTT Energy Decay Verification:")
    print("Layer  Actual E    Predicted E   Ratio")
    print("-" * 40)
    for d in range(0, solver.L, 4):
        print(f"{d:3d}    {energy[d]:.6f}    {predicted[d]:.6f}    {energy[d]/predicted[d]:.6f}")
