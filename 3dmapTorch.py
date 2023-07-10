import torch
import numpy as np
import matplotlib.pyplot as plt
import pywt
import multiprocessing as mp
import sys,os

def process_f(f, x, XF, M, t, params):
    r, m, p, k = params

    win = ((f**r) / (torch.sqrt(torch.tensor(2 * np.pi)) * (m * (f)**p + k))) * torch.exp(-0.5 * (1 / ((m * (f)**p + k)**2)) * (f**(2 * r)) * t**2)

    win = win / torch.sum(win)  # normalization
    W = torch.fft.fft(win)
    S_matrix_f = torch.fft.ifft(XF[f:f + M] * W)

    return S_matrix_f

def Proposed_ST(x, fe, fmax, params):
    M = len(x)
    if M % 2 != 0:
        t = torch.flip(torch.cat((torch.arange(1, torch.floor_divide(M, 2) + 1), torch.arange(-torch.floor_divide(M, 2), 1))), dims=[0]) / M
    else:
        t = torch.flip(torch.cat((torch.arange(1, torch.floor_divide(M, 2) + 1), torch.arange(-torch.floor_divide(M, 2) + 1, 1))), dims=[0]) / M

    Fx = torch.fft.fft(x.clone().detach())

    XF = torch.cat((Fx, Fx))

    Pas = int(fmax * M / fe)

    nResult = []
    for f in range(1, Pas + 1):
        nResult.append(process_f(f, x, XF, M, t, params))

    S_matrix = torch.vstack(nResult)  # Convert the list of arrays to a 2D array

    N_S_matrix = S_matrix / torch.sqrt(torch.sum(torch.sum(S_matrix * torch.conj(S_matrix))))
    CM_proposed = 1 / torch.sum(torch.sum(torch.abs(N_S_matrix)))

    y = S_matrix

    return y, CM_proposed

def evaluate_params(paramList):
    sig, fe, fmax, params = paramList

    r = params[0]
    m = params[1]
    p = params[2]
    k = params[3]

    params = [r, m, p, k]
    result, CM_proposed = Proposed_ST(sig, fe, fmax, params)
    score = CM_proposed

    print("Worker {} reporting score {} with params {}".format(os.getpid(), score, params))

    return score, params

# パラメータ探索を並列で実行
if __name__ == '__main__':
    dt = 0.001  # タイムステップ
    fs = 1 / dt  # サンプリング周波数
    nq_f = fs / 2.0  # ナイキスト周波数
    t = torch.arange(0, 1, dt)  # 時間配列
    N = t.size()  # サンプル数

    freqs = torch.linspace(1, nq_f, 500)  # 周波数配列

    # シグナルの定義
    sig = torch.tensor(
        pywt.data.demo_signal('WernerSorrows', t.size()[0])
    )
    
    fe = fs  # サンプリング周波数
    fmax = nq_f  # 最大周波数
    param_range = {
        'r': torch.arange(1.0, 0.5, -0.1),
        'm': torch.arange(0.1, 1.0, 0.1),
        'p': torch.arange(0.1, 1.0, 0.1),
        'k': torch.arange(0.1, 1.0, 0.1)
    }

    best_params = None
    best_score = -np.inf

    results = []

    paramList = []
    print("Building Params!")
    for r in param_range['r']:
        for m in param_range['m']:
            for p in param_range['p']:
                for k in param_range['k']:
                    paramList.append((sig, fe, fmax, (r, m, p, k)))

    mp.set_start_method('fork')

    print("Feeding a pool (strategy = {}) the pool with {} workers".format(mp.get_start_method(), len(paramList)))
    with mp.Pool(processes=10) as pool:
        results = [_ for _ in pool.imap_unordered(evaluate_params, paramList, chunksize=10)]

    cm_values = []
    params_list = []

    for score, params in results:
        if score > best_score:
            best_score = score
            best_params = params
        cm_values.append(score)
        params_list.append(params)

    print('Best Score: {}, Best Params: {}'.format(best_score, best_params))

    params5 = best_params

    result5, CM_proposed = Proposed_ST(sig, fe, fmax, params5)

    result5 = result5.numpy()

    r_values = [params[0] for params in params_list]
    m_values = [params[1] for params in params_list]
    p_values = [params[2] for params in params_list]
    k_values = [params[3] for params in params_list]

    for r in param_range['r']:
        cm_values_r = [cm_values[i] for i in range(len(cm_values)) if params_list[i][0] == r]
        m_values_r = [params_list[i][1] for i in range(len(params_list)) if params_list[i][0] == r]
        p_values_r = [params_list[i][2] for i in range(len(params_list)) if params_list[i][0] == r]
        k_values_r = [params_list[i][3] for i in range(len(params_list)) if params_list[i][0] == r]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(m_values_r, p_values_r, k_values_r, c=cm_values_r, cmap='jet')
        ax.set_xlabel('m')
        ax.set_ylabel('p')
        ax.set_zlabel('k')
        ax.set_title(f'CM_proposed (r = {r})')

    for r in param_range['r']:
        cm_values_r = [cm_values[i] for i in range(len(cm_values)) if params_list[i][0] == r]
        plt.figure()
        plt.plot(cm_values_r)

        plt.xlabel('Iteration')
        plt.ylabel('CM_proposed')

    plt.figure()
    plt.xlabel("Time[s]")
    plt.ylabel("Signal")
    plt.plot(sig)

    plt.figure()
    plt.imshow(np.abs(result5), extent=[0, 1, 0, fmax], aspect='auto', cmap='jet', origin='lower')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(label='Magnitude')
    title = f"r = {params5[0]}, m = {params5[1]}, p = {params5[2]}, k = {params5[3]}"
    plt.title(title)

    plt.figure()
    plt.plot(cm_values)
    plt.xlabel('Iteration')
    plt.ylabel('CM_proposed')

    plt.subplot()

    plt.show()