import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings("ignore")

import cv2
from scipy.interpolate import interp1d

def get_cat_points_from_image(filename='cat.png', N=256, use_canny=False):
    """
    Extract cat outline from PNG. For line art, set use_canny=True.
    Uses CHAIN_APPROX_NONE for dense points.
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image {filename} not found. Ensure it's a silhouette PNG.")
    print(f"Image loaded: shape {img.shape}, min/max values {img.min()}/{img.max()}")
    
    # Save original for debug
    cv2.imwrite('original_thresh_debug.png', img)
    print("Saved 'original_thresh_debug.png' - inspect image values")
    
    if use_canny:
        edges = cv2.Canny(img, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        _, thresh = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
    else:
        # For white cat on black BG: THRESH_BINARY keeps white=255, black=0
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Save thresh for debug
    cv2.imwrite('thresh.png', thresh)
    print("Saved 'thresh.png' - cat should be white (255) on black (0)")
    
    # Use RETR_EXTERNAL + CHAIN_APPROX_NONE for full boundary
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"Found {len(contours)} contours")
    if not contours:
        raise ValueError("No contours. Lower threshold (e.g., 100) or use_canny=True.")
    
    # Largest contour (cat boundary)
    cont = max(contours, key=cv2.contourArea)
    print(f"Largest contour points: {len(cont)}")
    
    points = cont.squeeze()
    print(f"After squeeze: {len(points)}")
    x, y = points[:, 1], points[:, 0]  # Swap to (x,y)
    # Center & scale
    x = (x - np.mean(x)) / np.std(x) * 2
    y = (y - np.mean(y)) / np.std(y) * 2
    z = x + 1j * y
    orig_len = len(z)
    print(f"After scaling: {orig_len} points")
    if orig_len < 10:
        raise ValueError("Too few points; try threshold=100.")
    
    # Resample
    z_extended = np.concatenate([z[-3:], z, z[:3]])
    t_extended = np.linspace(0, 1, len(z_extended))
    t_new = np.linspace(0, 1, N)
    interp_real = interp1d(t_extended, z_extended.real, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interp_imag = interp1d(t_extended, z_extended.imag, kind='cubic', bounds_error=False, fill_value='extrapolate')
    z_resampled = interp_real(t_new) + 1j * interp_imag(t_new)
    z_resampled[-1] = z_resampled[0]
    
    # Debug plot
    fig_raw, ax_raw = plt.subplots(figsize=(8,6))
    ax_raw.plot(x, y, 'r-o', markersize=1, label=f'Raw ({orig_len} pts)')
    ax_raw.plot(z_resampled.real, z_resampled.imag, 'b-', linewidth=2, label=f'Resampled ({N} pts)')
    ax_raw.set_title('Raw vs Resampled Contour (Debug)')
    ax_raw.axis('equal')
    ax_raw.legend()
    ax_raw.grid(True)
    plt.savefig('raw_contour.png', dpi=150, bbox_inches='tight')
    plt.close(fig_raw)
    print("Debug plot saved: 'raw_contour.png' (should be cat-shaped now!)")
    
    return z_resampled

# [Unchanged: compute_dft_coefficients, etc.]
def compute_dft_coefficients(z):
    N = len(z)
    X = np.fft.fft(z) / N
    return X

def get_epicycle_params(X, sort_by_magnitude=True):
    N = len(X)
    omega = np.arange(N)
    if N > 1:
        omega[N//2:] -= N
    R = np.abs(X)
    phi = np.angle(X)
    if sort_by_magnitude:
        idx = np.argsort(R)[::-1]
        return R[idx], phi[idx], omega[idx], idx
    else:
        return R, phi, omega, np.arange(N)

def reconstruct_curve(X, t, M=None):
    N = len(X)
    z_recon = np.zeros_like(t, dtype=complex)
    if M is None:
        M = N // 2
    z_recon += X[0]
    for kk in range(1, min(M, N//2)):
        exp_pos = np.exp(1j * kk * t)
        exp_neg = np.exp(-1j * kk * t)
        z_recon += X[kk] * exp_pos + X[N - kk] * exp_neg
    if N % 2 == 0 and M >= N//2:
        nyq = N // 2
        z_recon += X[nyq] * np.exp(1j * nyq * t)
    return z_recon

def animate_epicycles(z, X, M=30, frames=100):
    N = len(z)
    R, phi, omega, _ = get_epicycle_params(X, sort_by_magnitude=True)
    R = R[:M]
    phi = phi[:M]
    omega = omega[:M]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Epicycles Tracing Cat Silhouette using DFT")
    
    margin = 0.2
    x_min, x_max = z.real.min() - margin, z.real.max() + margin
    y_min, y_max = z.imag.min() - margin, z.imag.max() + margin
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_aspect('equal')
    ax1.grid(True)
    line, = ax1.plot([], [], 'b-', lw=2, label='Trace')
    point, = ax1.plot([], [], 'ro', markersize=10)
    circles = [Circle((0,0), r, fill=False, color=plt.cm.viridis(i/M)) for i, r in enumerate(R)]
    for circ in circles:
        ax1.add_patch(circ)
    lines = [ax1.plot([], [], 'k-', lw=1)[0] for _ in range(M)]
    
    t_full = np.linspace(0, 2*np.pi, 1000)
    z_full = reconstruct_curve(X, t_full)
    ax2.plot(z_full.real, z_full.imag, 'g--', alpha=0.5, label='Full Reconstructed Curve')
    ax2.plot(z.real, z.imag, 'bo', markersize=2, label='Sample Points')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.legend()
    
    def update(frame):
        t = (frame / frames) * 2 * np.pi
        current_pos = 0j
        for i in range(M):
            angle = omega[i] * t + phi[i]
            dx = R[i] * np.cos(angle)
            dy = R[i] * np.sin(angle)
            prev_pos = current_pos
            current_pos += complex(dx, dy)
            circles[i].center = (prev_pos.real, prev_pos.imag)
            lines[i].set_data([prev_pos.real, current_pos.real], [prev_pos.imag, current_pos.imag])
        
        trace_steps = min(100, frames)
        trace_t = np.linspace(0, t, trace_steps)
        trace_z = np.sum([R[j] * np.exp(1j * (omega[j] * trace_t + phi[j])) for j in range(M)], axis=0)
        line.set_data(trace_z.real, trace_z.imag)
        point.set_data([current_pos.real], [current_pos.imag])
        
        return [line, point] + lines + [c for c in circles]
    
    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False, repeat=True)
    ani.save('cat_epicycles.gif', writer='pillow', fps=10)
    plt.close(fig)
    print("Animation saved as 'cat_epicycles.gif'")
    return ani

if __name__ == "__main__":
    N = 256
    try:
        z = get_cat_points_from_image('cat.png', N, use_canny=False)
    except Exception as e:
        print(f"Extraction error: {e}")
        print("Try: Change threshold to 100 in cv2.threshold, or use_canny=True.")
        exit(1)
    
    X = compute_dft_coefficients(z)
    
    fig1, (ax_freq, ax_orig) = plt.subplots(1, 2, figsize=(10, 4))
    R, _, omega, _ = get_epicycle_params(X)
    ax_freq.semilogy(np.abs(omega), R, 'o-')
    ax_freq.set_title('Magnitude Spectrum')
    ax_freq.set_xlabel('Frequency k')
    ax_freq.set_ylabel('|X_k|')
    ax_freq.grid(True)
    
    margin = 0.2
    x_min, x_max = z.real.min() - margin, z.real.max() + margin
    y_min, y_max = z.imag.min() - margin, z.imag.max() + margin
    ax_orig.plot(z.real, z.imag, 'bo-', label='Cat Outline')
    ax_orig.set_xlim(x_min, x_max)
    ax_orig.set_ylim(y_min, y_max)
    ax_orig.axis('equal')
    ax_orig.set_title('Original Sampled Points for Cat')
    ax_orig.grid(True)
    ax_orig.legend()
    
    plt.tight_layout()
    plt.savefig('cat_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Static plots saved as 'cat_plots.png'")
    
    ani = animate_epicycles(z, X, M=30, frames=100)