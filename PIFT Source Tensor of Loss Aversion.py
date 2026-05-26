import numpy as np
import matplotlib.pyplot as plt
import os

# หาตำแหน่ง Absolute Path ของไฟล์สคริปต์นี้โดยอัตโนมัติ เพื่อป้องกันปัญหา Read-only File System
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

# 1. กำหนดค่าพารามิเตอร์มาตรฐานจากงานวิจัย (Kahneman & Tversky, 1992)
alpha = 0.88  # เลขชี้กำลังฝั่งกำไร
beta = 0.88   # เลขชี้กำลังฝั่งขาดทุน
lambda_behavioral = 2.25  # อัตราส่วนความกลัวความสูญเสียระดับพฤติกรรม

# 2. สร้างช่วงพิกัดพฤติกรรม Delta
delta_gains = np.linspace(1e-5, 2.0, 1000)
delta_losses = np.linspace(-1e-5, -2.0, 1000)

# ==========================================
# 3. คำนวณค่าทางคณิตศาสตร์และเรขาคณิต (แก้ไขส่วนกำลัง)
# ==========================================

# คำนวณ Metric & Source Tensor: g11 = T11 = (dV_KT / dDelta)^2
g11_gains = (alpha * (delta_gains ** (alpha - 1))) ** 2
g11_losses = (lambda_behavioral * beta * ((-delta_losses) ** (beta - 1))) ** 2

# คำนวณ Geodesic Distance L(Delta)
l_gains = np.zeros_like(delta_gains)
sqrt_g11_gains = np.sqrt(g11_gains)
for i in range(1, len(delta_gains)):
    l_gains[i] = np.trapezoid(sqrt_g11_gains[:i+1], delta_gains[:i+1])

l_losses = np.zeros_like(delta_losses)
sqrt_g11_losses = np.sqrt(g11_losses)
for i in range(1, len(delta_losses)):
    l_losses[i] = -np.trapezoid(sqrt_g11_losses[:i+1], -delta_losses[:i+1])

# แก้ไขปัญหา RuntimeWarning: แยกการคำนวณเลขชี้กำลังไม่ให้ปนเปื้อนแดนลบ
v_kt_gains = delta_gains ** alpha
v_kt_losses = -lambda_behavioral * ((-delta_losses) ** beta)

# รวมพิกัดเรียงจากซ้ายไปขวา (-2 ไปหา +2)
delta_total = np.concatenate((delta_losses[::-1], delta_gains))
g11_total = np.concatenate((g11_losses[::-1], g11_gains))
gamma111_total = (alpha - 1) / delta_total

# รวมฟังก์ชันมูลค่าและระยะทางที่กู้คืนมาได้
v_kt_total = np.concatenate((v_kt_losses[::-1], v_kt_gains))
l_geodesic_total = np.concatenate((l_losses[::-1], l_gains))

# ตั้งค่าฟอนต์มาตรฐานสำหรับเล่มรายงานวิจัย
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# ==========================================
# รูปที่ 1: Metric & Source Tensor (g11 = T11)
# ==========================================
plt.figure(figsize=(6, 5))
plt.plot(delta_total, g11_total, color='darkblue', linewidth=2.5, label=r'$g_{11}(\Delta) = T_{11}(\Delta)$')
plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
plt.title("Behavioral Space Metric & Source Tensor", fontweight='bold', pad=15)
plt.xlabel(r"Behavioral Coordinate ($\Delta$)")
plt.ylabel(r"Metric Component Value ($g_{11}$)")
plt.ylim(0, 15)  
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.text(0.4, 8, r'Gains: $\alpha^2 \Delta^{2(\alpha-1)}$', fontsize=10, color='green')
plt.text(-1.8, 8, r'Losses: $\lambda^2 \beta^2 (-\Delta)^{2(\beta-1)}$', fontsize=10, color='red')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'figure1_metric_source.png'), dpi=300)
plt.show()

# ==========================================
# รูปที่ 2: Christoffel Symbol (Geometric Connection)
# ==========================================
plt.figure(figsize=(6, 5))
plt.plot(delta_total, gamma111_total, color='crimson', linewidth=2.5, label=r'$\Gamma^1_{11}(\Delta)$')
plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
plt.axhline(0, color='gray', linestyle='-', alpha=0.5)
plt.title("Geometric Connection & Cognitive Pull", fontweight='bold', pad=15)
plt.xlabel(r"Behavioral Coordinate ($\Delta$)")
plt.ylabel(r"Christoffel Symbol Value ($\Gamma^1_{11}$)")
plt.ylim(-10, 10)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'figure2_christoffel.png'), dpi=300)
plt.show()

# ==========================================
# รูปที่ 3: Geodesic Metric Recovery Verification
# ==========================================
plt.figure(figsize=(6, 5))
# พล็อตฟังก์ชันคลาสสิก Kahneman-Tversky ดั้งเดิม
plt.plot(delta_total, v_kt_total, 'o', color='black', alpha=0.3, markersize=5, label='Empirical KT Value Function')
# พล็อตเส้นระยะทางจีโอเดสิก (แก้ Raw string ป้องกัน SyntaxWarning)
plt.plot(delta_total, l_geodesic_total, color='darkgreen', linewidth=2.5, label=r'Recovered Geodesic Distance $L(\Delta)$')
plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
plt.axhline(0, color='gray', linestyle='-', alpha=0.5)
plt.title("Exact Geodesic Distance Recovery", fontweight='bold', pad=15)
plt.xlabel(r"Behavioral Coordinate ($\Delta$)")
plt.ylabel(r"Psychological Value / Length ($L$)")
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'figure3_geodesic_recovery.png'), dpi=300)
plt.show()
