import numpy as np
import matplotlib.pyplot as plt

# 1. Configuration & Mathematical Constants (Kahneman-Tversky standard parameters)
alpha = 0.88
lam = 2.25

delta = np.linspace(-2, 2, 1000)

# Prevent computing negative bases to a fractional power (Fix RuntimeWarning)
delta_pos = np.clip(delta, 0, None)
delta_neg = np.clip(-delta, 0, None)

# Kahneman-Tversky Value Function
V_KT = np.where(delta >= 0, delta_pos**alpha, -lam * (delta_neg)**alpha)

# Under kappa = -1, the behavioral potential directly matches the value function: \Phi_G = V_KT
Phi_G = V_KT  

# Slicing safely for post-derivative elements to handle the boundary singularity
delta_gain = delta[delta > 0]
delta_loss = delta[delta < 0]

# 2. Geometric Quantity Calculations (Taking kappa = -1 into account)
# Cognitive Pull (Force) remains restorative toward the reference point
pull_gain = alpha * (delta_gain)**(alpha - 1)
pull_loss = lam * alpha * (-delta_loss)**(alpha - 1)

# Psychological Ricci Curvature R_00
curv_gain = alpha * (1 - alpha) * (delta_gain)**(alpha - 2)
curv_loss = lam * alpha * (1 - alpha) * (-delta_loss)**(alpha - 2)

# --- FIGURE 1: Value Function & Potential (Direct Match) ---
plt.figure(figsize=(6, 4.5))
plt.plot(delta, V_KT, 'b-', linewidth=2, label=r'$V_{KT}(\Delta)$ (Kahneman)')
plt.plot(delta, Phi_G, 'r--', linewidth=2, label=r'$\Phi_G(\Delta)$ (Potential under $\kappa=-1$)')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('1. Value Function and Behavioral Potential Alignment')
plt.xlabel(r'$\Delta$ (Outcomes)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig('metric_1_potential.png', dpi=150)
plt.close()

# --- FIGURE 2: Cognitive Pull ---
plt.figure(figsize=(6, 4.5))
plt.plot(delta_gain, pull_gain, 'g-', label='Gain Domain Pull')
plt.plot(delta_loss, pull_loss, 'r-', label='Loss Domain Pull')
plt.axvline(0, color='black', linewidth=0.5)
plt.title(r'2. Cognitive Pull ($\Gamma^1_{00}$)')
plt.xlabel(r'$\Delta$')
plt.ylabel('Force Magnitude')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig('metric_2_pull.png', dpi=150)
plt.close()

# --- FIGURE 3: Psychological Ricci Curvature ---
plt.figure(figsize=(6, 4.5))
plt.plot(delta_gain, curv_gain, 'g-', label='Gain Curvature')
plt.plot(delta_loss, curv_loss, 'r-', label='Loss Curvature')
plt.yscale('log')
plt.axvline(0, color='black', linewidth=0.5)
plt.title(r'3. Psychological Ricci Curvature ($R_{00}$)')
plt.xlabel(r'$\Delta$')
plt.ylabel('Curvature (Log Scale)')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig('metric_3_curvature.png', dpi=150)
plt.close()

# --- FIGURE 4: Asymmetry Loss Aversion Ratio ---
plt.figure(figsize=(6, 4.5))
delta_test = np.linspace(0.01, 2.0, 100)
pull_ratio = (lam * alpha * (delta_test)**(alpha - 1)) / (alpha * (delta_test)**(alpha - 1))
plt.plot(delta_test, pull_ratio, 'k-', label='Pull Ratio (Loss/Gain)')
plt.axhline(2.25, color='magenta', linestyle='--', label=r'Kahneman $\lambda = 2.25$')
plt.title('4. Emergent Loss Aversion Ratio')
plt.xlabel(r'$|\Delta|$')
plt.ylabel('Ratio')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig('metric_4_ratio.png', dpi=150)
plt.close()

print("All 4 figures updated and saved successfully using the kappa = -1 paradigm.")
