import librosa, torch, numpy as np, os, scipy.io as sio
# import numpy as np

fs = 16000
def singleSrcSTFT(m1_path, m2_path):
    m1, _ = librosa.load(m1_path, sr=fs)
    m2, _ = librosa.load(m2_path, sr=fs)

    nFFT = 320  # 20ms
    win_size = nFFT
    shift = int(win_size / 2)
    win_type = 'hamming'

    stft_1 = librosa.stft(m1, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
    stft_2 = librosa.stft(m2, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)

    stft_1_mag = torch.FloatTensor(np.abs(stft_1))
    stft_2_mag = torch.FloatTensor(np.abs(stft_2))
    stft_1_phs = torch.FloatTensor(np.angle(stft_1))
    stft_2_phs = torch.FloatTensor(np.angle(stft_2))

    mag_multi_20ms = torch.cat((stft_1_mag.unsqueeze(0), stft_2_mag.unsqueeze(0)), dim=0)
    phs_multi_20ms = torch.cat((stft_1_phs.unsqueeze(0), stft_2_phs.unsqueeze(0)), dim=0)

    id = os.path.split(m1_path)[-1][:-7]
    sio.savemat('stft/{}_mag_20ms.mat'.format(id), {'mag': mag_multi_20ms.numpy()})
    sio.savemat('stft/{}_phs_20ms.mat'.format(id), {'phs': phs_multi_20ms.numpy()})


if __name__ == '__main__':
    m1_path = 'audio/mic/84-121123-0000_m1.wav'
    m2_path = 'audio/mic/84-121123-0000_m2.wav'
    singleSrcSTFT(m1_path, m2_path)
