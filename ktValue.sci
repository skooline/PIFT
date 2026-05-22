// =============================================================================
// PIFT geodesic vs KT value function (Separate Windows Plot with Fixed Grids)
// =============================================================================

clear;
clc;

// --- 1. กำหนดพารามิเตอร์เริ่มต้น ---
alpha_val = 0.88;    // KT alpha
lambda_val = 2.25;   // KT lambda
N = 600;             // จำนวนจุดในการคำนวณ

// --- 2. ฟังก์ชันคำนวณพื้นฐาน ---

function v = ktValue(x, a, lam)
    v = zeros(x);
    for i = 1:length(x)
        if x(i) >= 0 then
            v(i) = max(x(i), 1e-12)^a;
        else
            v(i) = -lam * ((-x(i))^a);
        end
    end
endfunction

function dv = dKT(x, a, lam)
    dv = zeros(x);
    for i = 1:length(x)
        if x(i) > 0 then
            dv(i) = a * (x(i)^(a - 1));
        elseif x(i) < 0 then
            dv(i) = lam * a * ((-x(i))^(a - 1));
        else
            dv(i) = 0;
        end
    end
endfunction

function g = g11exact(x, a, lam)
    dv = dKT(x, a, lam);
    g = dv .^ 2;
endfunction

function out = cumulTrapz(xs, fs)
    n = length(xs);
    out = zeros(1, n);
    for i = 2:n
        out(i) = out(i-1) + 0.5 * (fs(i-1) + fs(i)) * (xs(i) - xs(i-1));
    end
endfunction


// --- 3. เริ่มการคำนวณหลัก ---

DA_neg = linspace(-3, -1e-4, N/2);
DA_pos = linspace(1e-4, 3, N/2);
DA_main = [DA_neg, DA_pos];

ktArr = ktValue(DA_main, alpha_val, lambda_val);
sqrtG = sqrt(g11exact(DA_main, alpha_val, lambda_val));

L_raw = cumulTrapz(DA_main, sqrtG);

iMid = length(DA_neg);
L_off = L_raw(iMid);
L_sign = zeros(DA_main);
for i = 1:length(DA_main)
    if DA_main(i) < 0 then
        L_sign(i) = -(L_off - L_raw(i));
    else
        L_sign(i) = L_raw(i) - L_off;
    end
end

pos_indices = find(DA_main >= 0);
max_kt_pos = max(ktArr(pos_indices));
max_L_pos = max(L_sign(pos_indices));
if max_L_pos == 0 then max_L_pos = 1; end
scalePos = max_kt_pos / max_L_pos;

L_norm = L_sign * scalePos;


// --- 4. การแสดงผลแยกหน้าต่างพร้อมเส้นตาราง ---

// รูปที่ 1: หน้าต่างคู่เทียบ KT กับ PIFT
scf(0); 
clf();
plot(DA_main, ktArr, "b-", "linewidth", 2.5);
plot(DA_main, L_norm, "r--", "linewidth", 2);
xtitle("PIFT Geodesic vs KT Value Function", "Delta", "Value");
hl = legend(["KT V(Delta)", "PIFT L(Delta) (Exact, No Reg)"], "in_lower_right");
xgrid(); // เปิดเส้นตารางแกนหลักและแกนรอง


// รูปที่ 2: หน้าต่างแสดง Metric g11 (Required Shape)
scf(1); 
clf();
DA_g = linspace(-2.5, 2.5, 400);
DA_g = DA_g(find(abs(DA_g) > 0.05)); 
G11 = g11exact(DA_g, alpha_val, lambda_val);

idx_pos = find(DA_g >= 0);
idx_neg = find(DA_g < 0);

plot(DA_g(idx_pos), G11(idx_pos), "g-", "linewidth", 2);
plot(DA_g(idx_neg), G11(idx_neg), "g-", "linewidth", 2);
xtitle("Metric g11(Delta) - Required Shape", "Delta", "g11(Delta)");
xgrid(); // เปิดเส้นตารางแกนหลักและแกนรอง


// รูปที่ 3: หน้าต่างแสดง Singularity ใกล้จุด 0 (แก้ไขจุดข้อความหลุดบรรทัดแล้ว)
scf(2); 
clf();
DA_near = linspace(-0.5, 0.5, 300);
DA_near = DA_near(find(abs(DA_near) > 1e-4));
G_near = g11exact(DA_near, alpha_val, lambda_val);

for i = 1:length(G_near)
    if G_near(i) > 50 then G_near(i) = 50; end
end

plot(DA_near, G_near, "m-", "linewidth", 2);
xtitle("Singularity at Delta=0 (Clipped at 50)", "Delta", "g11");
xgrid(); // เปิดเส้นตารางแกนหลักและแกนรอง
