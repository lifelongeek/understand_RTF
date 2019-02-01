import librosa
import numpy as np
import torch
import scipy.io as sio
import pdb
import math

fs = 16000

def get_pairwise_phase_mod(input):
    PI = math.pi
    TWOPI = PI*2
    nCH, F, T = input.size()
    nCombination = int(nCH * (nCH - 1) / 2)

    output = torch.FloatTensor(nCombination, F, T).zero_()

    count = 0
    for i in range(nCH):
        for j in range(i + 1, nCH):
            output[count, :, :] = torch.remainder(input[i, :, :] - input[j, :, :] + PI, TWOPI) - PI
            count += 1

    assert (count == nCombination)

    return output


# Case 1. #Src = 1, #Mic = 4, RT60 = 0
print('processing case 1')
y11, _ = librosa.load('audio/case1_y11.wav', sr=fs)
y12, _ = librosa.load('audio/case1_y12.wav', sr=fs)
y13, _ = librosa.load('audio/case1_y13.wav', sr=fs)
y14, _ = librosa.load('audio/case1_y14.wav', sr=fs)

nFFT = 320 # 20ms
win_size=nFFT
shift=int(win_size/2)
win_type='hamming'

stft_1 = librosa.stft(y11, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_2 = librosa.stft(y12, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_3 = librosa.stft(y13, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_4 = librosa.stft(y14, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)

stft_1_mag = torch.FloatTensor(np.abs(stft_1))
stft_2_mag = torch.FloatTensor(np.abs(stft_2))
stft_3_mag = torch.FloatTensor(np.abs(stft_3))
stft_4_mag = torch.FloatTensor(np.abs(stft_4))

stft_1_phs = torch.FloatTensor(np.angle(stft_1))
stft_2_phs = torch.FloatTensor(np.angle(stft_2))
stft_3_phs = torch.FloatTensor(np.angle(stft_3))
stft_4_phs = torch.FloatTensor(np.angle(stft_4))

mag_multi_20ms = torch.cat(
    (stft_1_mag.unsqueeze(0), stft_2_mag.unsqueeze(0), stft_3_mag.unsqueeze(0), stft_4_mag.unsqueeze(0)), dim=0)
phs_multi_20ms = torch.cat(
    (stft_1_phs.unsqueeze(0), stft_2_phs.unsqueeze(0), stft_3_phs.unsqueeze(0), stft_4_phs.unsqueeze(0)), dim=0)

# pmd_20ms = get_pairwise_magnitude(mag_multi_20ms, c=0.01)
# ppd_20ms_wrap = get_pairwise_phase(phs_multi_20ms)
ppd_20ms_wrap_mod = get_pairwise_phase_mod(phs_multi_20ms)

#pdb.set_trace()

sio.savemat('stft/case1_mag_20ms.mat', {'mag':mag_multi_20ms.numpy()})
sio.savemat('stft/case1_phs_20ms.mat', {'phs':phs_multi_20ms.numpy()})
sio.savemat('stft/case1_ppd_20ms_wrap_mod.mat', {'ppd_wrap_mod':ppd_20ms_wrap_mod.numpy()})



nFFT = 8000 # 500ms
win_size=nFFT
shift = int(win_size/2)
win_type='hamming'

stft_1 = librosa.stft(y11, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_2= librosa.stft(y12, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_3= librosa.stft(y13, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_4= librosa.stft(y14, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)

stft_1_mag = torch.FloatTensor(np.abs(stft_1))
stft_2_mag = torch.FloatTensor(np.abs(stft_2))
stft_3_mag = torch.FloatTensor(np.abs(stft_3))
stft_4_mag = torch.FloatTensor(np.abs(stft_4))

stft_1_phs = torch.FloatTensor(np.angle(stft_1))
stft_2_phs = torch.FloatTensor(np.angle(stft_2))
stft_3_phs = torch.FloatTensor(np.angle(stft_3))
stft_4_phs = torch.FloatTensor(np.angle(stft_4))

mag_multi_500ms = torch.cat(
    (stft_1_mag.unsqueeze(0), stft_2_mag.unsqueeze(0), stft_3_mag.unsqueeze(0), stft_4_mag.unsqueeze(0)), dim=0)
phs_multi_500ms = torch.cat(
    (stft_1_phs.unsqueeze(0), stft_2_phs.unsqueeze(0), stft_3_phs.unsqueeze(0), stft_4_phs.unsqueeze(0)), dim=0)

ppd_500ms_wrap_mod = get_pairwise_phase_mod(phs_multi_500ms)

sio.savemat('stft/case1_mag_500ms.mat', {'mag':mag_multi_500ms.numpy()})
sio.savemat('stft/case1_phs_500ms.mat', {'phs':phs_multi_500ms.numpy()})
sio.savemat('stft/case1_ppd_500ms_wrap_mod.mat', {'ppd_wrap_mod':ppd_500ms_wrap_mod.numpy()})


# Case3. Src = 2, #Mic = 4, RT60 = 0
print('processing case 3')
y11, _ = librosa.load('audio/case1_y11.wav', sr=fs)
y12, _ = librosa.load('audio/case1_y12.wav', sr=fs)
y13, _ = librosa.load('audio/case1_y13.wav', sr=fs)
y14, _ = librosa.load('audio/case1_y14.wav', sr=fs)

y21, _ = librosa.load('audio/case3_y21.wav', sr=fs)
y22, _ = librosa.load('audio/case3_y22.wav', sr=fs)
y23, _ = librosa.load('audio/case3_y23.wav', sr=fs)
y24, _ = librosa.load('audio/case3_y24.wav', sr=fs)

y1, _ = librosa.load('audio/case3_y1.wav', sr=fs) # mixture
y2, _ = librosa.load('audio/case3_y2.wav', sr=fs) # mixture
y3, _ = librosa.load('audio/case3_y3.wav', sr=fs) # mixture
y4, _ = librosa.load('audio/case3_y4.wav', sr=fs) # mixture


nFFT = 320 # 20ms
win_size=nFFT
shift=int(win_size/2)
win_type='hamming'

stft_21 = librosa.stft(y21, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_22 = librosa.stft(y22, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_23 = librosa.stft(y23, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_24 = librosa.stft(y24, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)

stft_21_mag = torch.FloatTensor(np.abs(stft_21))
stft_22_mag = torch.FloatTensor(np.abs(stft_22))
stft_23_mag = torch.FloatTensor(np.abs(stft_23))
stft_24_mag = torch.FloatTensor(np.abs(stft_24))

stft_21_phs = torch.FloatTensor(np.angle(stft_21))
stft_22_phs = torch.FloatTensor(np.angle(stft_22))
stft_23_phs = torch.FloatTensor(np.angle(stft_23))
stft_24_phs = torch.FloatTensor(np.angle(stft_24))

mag_multi_s2_20ms = torch.cat(
    (stft_21_mag.unsqueeze(0), stft_22_mag.unsqueeze(0), stft_23_mag.unsqueeze(0), stft_24_mag.unsqueeze(0)), dim=0)
phs_multi_s2_20ms = torch.cat(
    (stft_21_phs.unsqueeze(0), stft_22_phs.unsqueeze(0), stft_23_phs.unsqueeze(0), stft_24_phs.unsqueeze(0)), dim=0)

ppd_s2_20ms_wrap_mod = get_pairwise_phase_mod(phs_multi_s2_20ms)


stft_1 = librosa.stft(y1, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_2 = librosa.stft(y2, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_3 = librosa.stft(y3, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_4 = librosa.stft(y4, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)

stft_1_mag = torch.FloatTensor(np.abs(stft_1))
stft_2_mag = torch.FloatTensor(np.abs(stft_2))
stft_3_mag = torch.FloatTensor(np.abs(stft_3))
stft_4_mag = torch.FloatTensor(np.abs(stft_4))

stft_1_phs = torch.FloatTensor(np.angle(stft_1))
stft_2_phs = torch.FloatTensor(np.angle(stft_2))
stft_3_phs = torch.FloatTensor(np.angle(stft_3))
stft_4_phs = torch.FloatTensor(np.angle(stft_4))

mag_multi_mix_20ms = torch.cat(
    (stft_1_mag.unsqueeze(0), stft_2_mag.unsqueeze(0), stft_3_mag.unsqueeze(0), stft_4_mag.unsqueeze(0)), dim=0)
phs_multi_mix_20ms = torch.cat(
    (stft_1_phs.unsqueeze(0), stft_2_phs.unsqueeze(0), stft_3_phs.unsqueeze(0), stft_4_phs.unsqueeze(0)), dim=0)

ppd_mix_20ms_wrap_mod = get_pairwise_phase_mod(phs_multi_mix_20ms)


sio.savemat('stft/case3_s2_mag_20ms.mat', {'mag':mag_multi_s2_20ms.numpy()})
sio.savemat('stft/case3_s2_phs_20ms.mat', {'phs':phs_multi_s2_20ms.numpy()})
sio.savemat('stft/case3_s2_ppd_20ms_wrap_mod.mat', {'ppd_wrap_mod':ppd_s2_20ms_wrap_mod.numpy()})

sio.savemat('stft/case3_mix_mag_20ms.mat', {'mag':mag_multi_mix_20ms.numpy()})
sio.savemat('stft/case3_mix_phs_20ms.mat', {'phs':phs_multi_mix_20ms.numpy()})
sio.savemat('stft/case3_mix_ppd_20ms_wrap_mod.mat', {'ppd_wrap_mod':ppd_mix_20ms_wrap_mod.numpy()})



nFFT = 8000 # 500ms
win_size=nFFT
shift=int(win_size/2)
win_type='hamming'

stft_21 = librosa.stft(y21, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_22 = librosa.stft(y22, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_23 = librosa.stft(y23, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_24 = librosa.stft(y24, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)

stft_21_mag = torch.FloatTensor(np.abs(stft_21))
stft_22_mag = torch.FloatTensor(np.abs(stft_22))
stft_23_mag = torch.FloatTensor(np.abs(stft_23))
stft_24_mag = torch.FloatTensor(np.abs(stft_24))

stft_21_phs = torch.FloatTensor(np.angle(stft_21))
stft_22_phs = torch.FloatTensor(np.angle(stft_22))
stft_23_phs = torch.FloatTensor(np.angle(stft_23))
stft_24_phs = torch.FloatTensor(np.angle(stft_24))

mag_multi_s2_500ms = torch.cat(
    (stft_21_mag.unsqueeze(0), stft_22_mag.unsqueeze(0), stft_23_mag.unsqueeze(0), stft_24_mag.unsqueeze(0)), dim=0)
phs_multi_s2_500ms = torch.cat(
    (stft_21_phs.unsqueeze(0), stft_22_phs.unsqueeze(0), stft_23_phs.unsqueeze(0), stft_24_phs.unsqueeze(0)), dim=0)

# pmd_s2_500ms = get_pairwise_magnitude(mag_multi_s2_500ms, c=0.01)
ppd_s2_500ms_wrap_mod = get_pairwise_phase_mod(phs_multi_s2_500ms)


stft_1 = librosa.stft(y1, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_2 = librosa.stft(y2, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_3 = librosa.stft(y3, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)
stft_4 = librosa.stft(y4, n_fft=nFFT, hop_length=shift, win_length=win_size, window=win_type)

stft_1_mag = torch.FloatTensor(np.abs(stft_1))
stft_2_mag = torch.FloatTensor(np.abs(stft_2))
stft_3_mag = torch.FloatTensor(np.abs(stft_3))
stft_4_mag = torch.FloatTensor(np.abs(stft_4))

stft_1_phs = torch.FloatTensor(np.angle(stft_1))
stft_2_phs = torch.FloatTensor(np.angle(stft_2))
stft_3_phs = torch.FloatTensor(np.angle(stft_3))
stft_4_phs = torch.FloatTensor(np.angle(stft_4))

mag_multi_mix_500ms = torch.cat(
    (stft_1_mag.unsqueeze(0), stft_2_mag.unsqueeze(0), stft_3_mag.unsqueeze(0), stft_4_mag.unsqueeze(0)), dim=0)
phs_multi_mix_500ms = torch.cat(
    (stft_1_phs.unsqueeze(0), stft_2_phs.unsqueeze(0), stft_3_phs.unsqueeze(0), stft_4_phs.unsqueeze(0)), dim=0)

# pmd_mix_500ms = get_pairwise_magnitude(mag_multi_mix_500ms, c=0.01)
ppd_mix_500ms_wrap_mod = get_pairwise_phase_mod(phs_multi_mix_500ms)


sio.savemat('stft/case3_s2_mag_500ms.mat', {'mag':mag_multi_s2_500ms.numpy()})
sio.savemat('stft/case3_s2_phs_500ms.mat', {'phs':phs_multi_s2_500ms.numpy()})
sio.savemat('stft/case3_s2_ppd_500ms_wrap_mod.mat', {'ppd_wrap_mod':ppd_s2_500ms_wrap_mod.numpy()})


sio.savemat('stft/case3_mix_mag_500ms.mat', {'mag':mag_multi_mix_500ms.numpy()})
sio.savemat('stft/case3_mix_phs_500ms.mat', {'phs':phs_multi_mix_500ms.numpy()})
sio.savemat('stft/case3_mix_ppd_500ms_wrap_mod.mat', {'ppd_wrap_mod':ppd_mix_500ms_wrap_mod.numpy()})