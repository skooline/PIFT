import numpy as np
import matplotlib.pyplot as plt

# --- 1. กำหนดพารามิเตอร์เริ่มต้น ---
alpha_val = 0.88
lambda_val = 2.25
N = 600

# --- 2. ฟังก์ชันคำนวณพื้นฐาน ---
def kt_value(x, a, lam):
    v = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] >= 0:
            v[i] = max(x[i], 1e-12) ** a
        else:
            v[i] = -lam * ((-x[i]) ** a)
    return v

def d_kt(x, a, lam):
    dv = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            dv[i] = a * (x[i] ** (a - 1))
        elif x[i] < 0:
            dv[i] = lam * a * ((-x[i]) ** (a - 1))
        else:
            dv[i] = 0
    return dv

def g11_exact(x, a, lam):
    return d_kt(x, a, lam) ** 2

# --- 3. เริ่มการคำนวณหลัก ---
DA_neg = np.linspace(-3, -1e-4, N // 2)
DA_pos = np.linspace(1e-4, 3, N // 2)
DA_main = np.concatenate([DA_neg, DA_pos])

kt_arr = kt_value(DA_main, alpha_val, lambda_val)
sqrt_g = np.sqrt(g11_exact(DA_main, alpha_val, lambda_val))

# Cumulative Integration โดยใช้ NumPy
L_raw = np.cumsum(0.5 * (sqrt_g[:-1] + sqrt_g[1:]) * np.diff(DA_main))
L_raw = np.insert(L_raw, 0, 0) # ใส่จุดเริ่มเป็น 0

i_mid = len(DA_neg)
L_off = L_raw[i_mid]
L_sign = np.zeros_like(DA_main)
for i in range(len(DA_main)):
    if DA_main[i] < 0:
        L_sign[i] = -(L_off - L_raw[i])
    else:
        L_sign[i] = L_raw[i] - L_off

pos_indices = np.where(DA_main >= 0)[0]
scale_pos = np.max(kt_arr[pos_indices]) / np.max(L_sign[pos_indices])
L_norm = L_sign * scale_pos

# --- 4. การแสดงผลกราฟแยกหน้าต่าง (ยึดหลักจดจำแกนและ Grid) ---

# รูปที่ 1: เปรียบเทียบ KT กับ PIFT
plt.figure(0)
plt.plot(DA_main, kt_arr, "b-", label="KT V(Delta)", linewidth=2.5)
plt.plot(DA_main, L_norm, "r--", label="PIFT L(Delta) (Exact)", linewidth=2)
plt.title("PIFT Geodesic vs KT Value Function")
plt.xlabel("Delta")
plt.ylabel("Value")
plt.legend(loc="lower right")
plt.grid(True, which='both', linestyle='--', linewidth=0.5) # เปิดแกนหลัก/รองแบบเส้นประ

# รูปที่ 2: Metric g11 (Required Shape)
plt.figure(1)
DA_g = np.linspace(-2.5, 2.5, 400)
DA_g = DA_g[np.abs(DA_g) > 0.05]
G11 = g11_exact(DA_g, alpha_val, lambda_val)
plt.plot(DA_g[DA_g >= 0], G11[DA_g >= 0], "g-", linewidth=2)
plt.plot(DA_g[DA_g < 0], G11[DA_g < 0], "g-", linewidth=2)
plt.title("Metric g11(Delta) - Required Shape")
plt.xlabel("Delta")
plt.ylabel("g11(Delta)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# รูปที่ 3: Singularity
plt.figure(2)
DA_near = np.linspace(-0.5, 0.5, 300)
DA_near = DA_near[np.abs(DA_near) > 1e-4]
G_near = g11_exact(DA_near, alpha_val, lambda_val)
G_near = np.clip(G_near, 0, 50) # ตัดยอดที่ 50
plt.plot(DA_near, G_near, "m-", linewidth=2)
plt.title("Singularity at Delta=0 (Clipped at 50)")
plt.xlabel("Delta")
plt.ylabel("g11")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# สั่งแสดงผลทุกหน้าต่างพร้อมกัน
plt.show()