import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from scipy.signal import chirp
from stockwell import st
import pywt

from joblib import Parallel, delayed

dt = 0.001  # タイムステップ
fs = 1 / dt  # サンプリング周波数
nq_f = fs / 2.0  # ナイキスト周波数
t = np.arange(0, 1, dt)  # 時間配列
N = t.size  # サンプル数

freqs = np.linspace(1, nq_f, 500)  # 周波数配列

# シグナルの定義
# sig = (np.exp(2*np.pi*1j*(N/2**2)*t) * ((0/8<=t) & (t<1/8)) +
#        np.exp(2*np.pi*1j*(N/2**3)*t) * ((1/8<=t) & (t<2/8)) +
#        np.exp(2*np.pi*1j*(N/2**4)*t) * ((2/8<=t) & (t<3/8)) +
#        np.exp(2*np.pi*1j*(N/2**5)*t) * ((3/8<=t) & (t<4/8)) +
#        np.exp(2*np.pi*1j*(N/2**6)*t) * ((4/8<=t) & (t<5/8)) +
#        np.exp(2*np.pi*1j*(-N/2**2)*t) * ((5/8<=t) & (t<6/8)) +
#        np.exp(2*np.pi*1j*(-N/2**3)*t) * ((6/8<=t) & (t<7/8)) +
#        np.exp(2*np.pi*1j*(-N/2**4)*t) * ((7/8<=t) & (t<8/8))).astype('float64')
# sig = pywt.data.demo_signal('MishMash', t.size)
# sig = pywt.data.demo_signal('WernerSorrows', t.size)
sig = pywt.data.demo_signal('HypChirps', t.size)

# ストックウェル変換
def Proposed_ST(x, fe, fmax, params):
    M = len(x)
    if M % 2 != 0:
        t = np.flip(np.concatenate((np.arange(1, np.floor(M/2) + 1), np.arange(-np.floor(M/2), 1)))) / M

    else:
        t = np.flip(np.concatenate((np.arange(1, np.floor(M/2) + 1), np.arange(-np.floor(M/2) + 1, 1)))) / M

    r = params[0]
    m = params[1]
    p = params[2]
    k = params[3]
    
    Fx = np.fft.fft(x)
    XF = np.concatenate((Fx, Fx))
    
    Pas = int(fmax * M / fe)
    
    S_matrix = np.zeros((Pas, M), dtype=complex)
    
    for f in range(1, Pas + 1):
        win = ((f**r) / (np.sqrt(2*np.pi) * (m*(f)**p + k))) * np.exp(-0.5 * (1 / ((m*(f)**p + k)**2)) * (f**(2*r)) * t**2)
        win = win / np.sum(win)  # normalization
        W = np.fft.fft(win)
        S_matrix[f - 1, :] = np.fft.ifft(XF[f:f+M] * W)
    
    N_S_matrix = S_matrix / np.sqrt(np.sum(np.sum(S_matrix * np.conj(S_matrix))))
    CM_proposed = 1 / np.sum(np.sum(np.abs(N_S_matrix)))
    
    y = S_matrix
    
    return y, CM_proposed

fe = fs  # サンプリング周波数
fmax = nq_f  # 最大周波数

params1 = [1,1,0,0]  # standardST
params2 = [0.8, 0, 1, 1] # Sejdic’s ST
params3 = [1, 0.05, 1, 0.1] # Assous’s ST
params4 = [0.6035, 0.3, 0.0386, 0.4276] # proposed ST

param_range = {
    'r': np.arange(0.001, 1.0, 0.001),
    'm': np.arange(0.001, 1.0, 0.001),
    'p': np.arange(0.001, 1.0, 0.001),
    'k': np.arange(0.001, 1.0, 0.001)
}

best_params = None
best_score = -np.inf

def evaluate_params(params):
    r = params[0]
    m = params[1]
    p = params[2]
    k = params[3]

    params = [r, m, p, k]
    result, CM_proposed = Proposed_ST(sig, fe, fmax, params)
    score = CM_proposed

    return score, params

# パラメータ探索を並列で実行
num_cores = 12  # 使用するコア数を指定
results = Parallel(n_jobs=num_cores, backend="threading")(
    delayed(evaluate_params)([r, m, p, k]) for r in param_range['r'] for m in param_range['m'] for p in param_range['p'] for k in param_range['k']
    )

# CM_proposedの値とパラメータの組を保存するリスト
cm_values = []
params_list = []

# 最適なスコアとそのパラメータを見つける
for score, params in results:
    if score > best_score:
        best_score = score
        best_params = params
    # CM_proposedの値とパラメータを保存
    cm_values.append(score)
    params_list.append(params)

params5 = best_params

result5, CM_proposed = Proposed_ST(sig, fe, fmax, params5)

# result5をndarrayに変換する
result5 = np.array(result5)

# 結果を表示
# 信号
plt.figure()
plt.xlabel("Time[s]")
plt.ylabel("Signal")
plt.plot(sig)

# パラメータを最適化したSTの結果
plt.figure()
plt.imshow(np.abs(result5), extent=[0, 1, 0, fmax], aspect='auto', cmap='jet', origin='lower')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar(label='Magnitude')
title = f"r = {params5[0]}, m = {params5[1]}, p = {params5[2]}, k = {params5[3]}"
plt.title(title)

# STの総和の推移
plt.figure()
plt.plot(cm_values)
plt.xlabel('Iteration')
plt.ylabel('CM_proposed')
plt.show()
