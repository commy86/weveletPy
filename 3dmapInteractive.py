import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pywt
import concurrent.futures
import multiprocessing as mp
import sys,os



def process_f(f, x, XF, M, t, params):
    r, m, p, k = params
    
    win = ((f**r) / (np.sqrt(2*np.pi) * (m*(f)**p + k))) * np.exp(-0.5 * (1 / ((m*(f)**p + k)**2)) * (f**(2*r)) * t**2)
    win = win / np.sum(win)  # normalization
    W = np.fft.fft(win)
    S_matrix_f = np.fft.ifft(XF[f:f+M] * W)
    
    return S_matrix_f

def Proposed_ST(x, fe, fmax, params):
    M = len(x)
    if M % 2 != 0:
        t = np.flip(np.concatenate((np.arange(1, np.floor(M/2) + 1), np.arange(-np.floor(M/2), 1)))) / M
    else:
        t = np.flip(np.concatenate((np.arange(1, np.floor(M/2) + 1), np.arange(-np.floor(M/2) + 1, 1)))) / M

    Fx = np.fft.fft(x)
    XF = np.concatenate((Fx, Fx))
    
    Pas = int(fmax * M / fe)
    
    # Use concurrent.futures to parallelize the loop
    #with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #    S_matrix_futures = [executor.submit(process_f, f, x, XF, M, t, params) for f in range(1, Pas + 1)]

    nResult=[]
    for f in range(1, Pas + 1):
        #print(f,x,XF,M,t)
        nResult.append( process_f(f, x, XF, M, t, params))
    
    #S_matrix = np.vstack([future.result() for future in S_matrix_futures])  # Convert the list of arrays to a 2D array
    S_matrix = np.vstack(nResult)  # Convert the list of arrays to a 2D array

    N_S_matrix = S_matrix / np.sqrt(np.sum(np.sum(S_matrix * np.conj(S_matrix))))
    CM_proposed = 1 / np.sum(np.sum(np.abs(N_S_matrix)))
    
    y = S_matrix
    
    return y, CM_proposed

#def evaluate_params(sig, fe, fmax, params):
def evaluate_params(paramList):

    sig, fe, fmax, params = paramList

    #print("Worker {}".format(params))

    r = params[0]
    m = params[1]
    p = params[2]
    k = params[3]

    params = [r, m, p, k]
    result, CM_proposed = Proposed_ST(sig, fe, fmax, params)
    score = CM_proposed

    print("Worker {} reporting score {} with params {}".format(os.getpid(),score,params))

    return score, params

# パラメータ探索を並列で実行
if __name__ == '__main__':

    dt = 0.001  # タイムステップ
    fs = 1 / dt  # サンプリング周波数
    nq_f = fs / 2.0  # ナイキスト周波数
    t = np.arange(0, 1, dt)  # 時間配列
    N = t.size  # サンプル数

    freqs = np.linspace(1, nq_f, 500)  # 周波数配列

    # シグナルの定義
    #sig = pywt.data.demo_signal('WernerSorrows', t.size)

    sig = (np.exp(2*np.pi*1j*(N/2**2)*t) * ((0/8<=t) & (t<1/8)) +
        np.exp(2*np.pi*1j*(N/2**3)*t) * ((1/8<=t) & (t<2/8)) +
        np.exp(2*np.pi*1j*(N/2**4)*t) * ((2/8<=t) & (t<3/8)) +
        np.exp(2*np.pi*1j*(N/2**5)*t) * ((3/8<=t) & (t<4/8)) +
        np.exp(2*np.pi*1j*(N/2**6)*t) * ((4/8<=t) & (t<5/8)) +
        np.exp(2*np.pi*1j*(-N/2**2)*t) * ((5/8<=t) & (t<6/8)) +
        np.exp(2*np.pi*1j*(-N/2**3)*t) * ((6/8<=t) & (t<7/8)) +
        np.exp(2*np.pi*1j*(-N/2**4)*t) * ((7/8<=t) & (t<8/8))).astype('float64')

    fe = fs  # サンプリング周波数
    fmax = nq_f  # 最大周波数
    param_range = {
        'r': np.arange(1.0, 0.5, -0.1),
        'm': np.arange(0.1, 1.0, 0.1),
        'p': np.arange(0.1, 1.0, 0.1),
        'k': np.arange(0.1, 1.0, 0.1)
    }

    best_params = None
    best_score = -np.inf

    results = []


    paramList=[]
    print("Building Params!")
    for r in param_range['r']:
        for m in param_range['m']:
            for p in param_range['p']:
                for k in param_range['k']:
                    paramList.append( (sig,fe,fmax,(r,m,p,k)) )


    ##if you use spawn on mac, try to force to use fork
    ##mp.set_start_method('fork')

    print("Feeding a pool (strategy = {}) the pool with {} workers".format(mp.get_start_method(),len(paramList)))
    with mp.Pool(processes=10) as pool:
        results = [_ for _ in pool.imap_unordered(evaluate_params , paramList,chunksize=10)]
        #for d in rpt:
        #    results = rpt

    ## with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    ##     for r in param_range['r']:
    ##         for m in param_range['m']:
    ##             for p in param_range['p']:
    ##                 for k in param_range['k']:
    ##                     params = [r, m, p, k]
    ##                     print(params)
    ##                     future = executor.submit(evaluate_params, sig, fe, fmax, params)
    ##                     results.append(future)

    # CM_proposedの値とパラメータの組を保存するリスト
    cm_values = []
    params_list = []

    #best_score = min([ score,_ for score,_ in results ])

    for score,params in results:
        if score > best_score:
            best_score = score
            best_params = params
        cm_values.append(score)
        params_list.append(params)


    print('Best Score: {}, Best Params: {}'.format(best_score,best_params))

    # 最適なスコアとそのパラメータを見つける
    ##for future in concurrent.futures.as_completed(results):
    ##    score, params = future.result()
    ##    if score > best_score:
    ##        best_score = score
    ##        best_params = params
    ##    # CM_proposedの値とパラメータを保存
    ##    cm_values.append(score)
    ##    params_list.append(params)

    params5 = best_params

    result5, CM_proposed = Proposed_ST(sig, fe, fmax, params5)

    # result5をndarrayに変換する
    result5 = np.array(result5)

    # 3Dグラフを作成
    r_values = [params[0] for params in params_list]
    m_values = [params[1] for params in params_list]
    p_values = [params[2] for params in params_list]
    k_values = [params[3] for params in params_list]

    # rごとに3Dグラフを作成
    for r in param_range['r']:
        cm_values_r = [cm_values[i] for i in range(len(cm_values)) if params_list[i][0] == r]
        m_values_r = [params_list[i][1] for i in range(len(params_list)) if params_list[i][0] == r]
        p_values_r = [params_list[i][2] for i in range(len(params_list)) if params_list[i][0] == r]
        k_values_r = [params_list[i][3] for i in range(len(params_list)) if params_list[i][0] == r]

        fig = go.Figure(data=[go.Scatter3d(
            x=m_values_r,
            y=p_values_r,
            z=k_values_r,
            mode='markers',
            marker=dict(
                size=5,
                color=cm_values_r,
                colorscale='Jet',
                #cmin=0.00175,
                #cmax=0.0038,
                opacity=0.8
            )
        )])

        fig.update_layout(
            scene=dict(
                xaxis=dict(title='m'),
                yaxis=dict(title='p'),
                zaxis=dict(title='k')
            ),
            title=f'CM_proposed (r = {r})',
            margin=dict(l=0, r=0, b=0, t=40),
            coloraxis_colorbar=dict(title='Colorbar Title')
        )

        fig.show()
    
    # rごとにグラフを作成
    for r in param_range['r']:
        cm_values_r = [cm_values[i] for i in range(len(cm_values)) if params_list[i][0] == r]
        plt.figure()
        plt.plot(cm_values_r)

        plt.xlabel('Iteration')
        plt.ylabel('CM_proposed')


    # 結果を表示
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
