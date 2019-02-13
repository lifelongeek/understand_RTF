import math
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import stats

def intermic_mag_diff(input):
    # input : torch tensor of shape (num batches, F, nCH, T) for fft magnitude
    # output : torch tensor for log(Mi/M0), for i = ids of mics other than 0th
    eps = 1e-8
    mag_thr = 1e-6
    minval = math.log(eps)
    maxval = 10000

    N, F, nCH, T = input.size()
    nCombination = nCH-1  # assume that reference mic appear at the first of dim

    # output = torch.FloatTensor(N, F, nCombination, T).zero_()
    PMD = torch.FloatTensor(N, nCombination, T).zero_()

    mask = (input[:, :, 0, :] > mag_thr)  # NxFxT
    mask = mask.float()
    unmasked_freq_count = torch.sum(mask, dim=1)  # NxT
    for i in range(nCH-1):
        pmd_masked = mask*input[:, :, i+1, :]/(input[:, :, 0, :] + eps)  # NxFxT
        pmd_masked_mean = torch.sum(pmd_masked, dim=1)/unmasked_freq_count  # NxT
        # pmd_masked_var = torch.sum(mask*(pmd_masked-pmd_masked_mean.expand_as(pmd_masked))**2, dim=1)/unmasked_freq_count
        # confidence = 1/pmd_masked_var  # NxT

        # indices = torch.argmax(confidence, dim=1)  # N
        PMD[:, i, :] = pmd_masked_mean
        # output[:, :, i, :] = input[:, :, i+1, :]/(input[:, :, 0, :] + eps)

    PMD = torch.log(PMD+eps)
    PMD = torch.clamp(PMD, min=minval, max=maxval)
    # output = torch.log(output+eps)
    # output = torch.clamp(output, min=minval, max=maxval)

    # return output
    return PMD


def intermic_phs_diff(input):
    # input: torch tensor of shape (num batches, F, nCH, T) for fft phase
    # output: torch tensor for unwrapped phase differences
    PI = math.pi
    TWOPI = PI*2
    N, F, nCH, T = input.size()
    nCombination = nCH-1 # assume that reference mic appear at the first of dim

    output = torch.FloatTensor(N, F, nCombination, T).zero_()

    for i in range(nCH-1):
        output[:, :, i, :] = input[:, :, i+1, :] - input[:, :, 0, :]

    # wrap
    output = torch.remainder(output + PI, TWOPI) - PI

    # unwrap
    output = output.numpy()
    output = np.unwrap(output, axis=1)
    output = torch.FloatTensor(output)

    return output


def frame_value_estimate(v, num_top=10, binWidth=0.05):
    # makes histogram of values in V, then finds average of values in top NUM_TOP bins.
    # Might be unused because simple mean calculation seems to be better option
    nbins = math.ceil((np.max(v)-np.min(v))/binWidth)
    hist, bin_edges = np.histogram(v, bins=nbins)
    v_bin_indices = np.digitize(v, bin_edges) - 1  # -1 bcs np.digitize output indices start from 1
    sorted_bins = np.flip(np.argsort(hist))  # bin numbers are sorted from tallest to shortest

    binSizeList = np.zeros(num_top)
    binMeanList = np.zeros(num_top)
    for i in range(num_top):
        mask = v_bin_indices == sorted_bins[i]
        binSize = mask.sum()
        binMean = np.sum(v*mask)/binSize
        binSizeList[i] = binSize
        binMeanList[i] = binMean

    estimate = np.sum(binMeanList*binSizeList)/np.sum(binSizeList)
    return estimate


def estimate_value(V, pair_id, name, num_top_bins=10, binWidth=0.05):
    # ppd - numpy array of shape (N, F, nPairs, T)
    # for now it only handles one pair
    # returns estimated value
    _, F, _, T = np.shape(V)
    tList = np.ceil(T * np.arange(start=0.05, stop=1, step=0.05)).astype(int)
    estimateList = np.zeros_like(tList).astype(float)

    confidenceList = np.zeros_like(tList).astype(float)
    for i in range(np.shape(tList)[0]):
        t = tList[i]
        Vframe = V[0, :, pair_id, t]  # unwrapped frame
        # diff_uwframe = np.diff(uwframe)
        # estimateList[i] = frame_value_estimate(Vframe, num_top_bins)
        estimateList[i] = np.mean(Vframe)
        # estimateList[i] = stats.trim_mean(Vframe, proportiontocut=0.2)
        confidenceList[i] = 1/np.std(Vframe)
    slope_estimate = estimateList[np.argmax(confidenceList)]

    plt.figure()
    x = np.arange(0.05, 1, 0.05)

    plt.subplot(211)
    plt.plot(x, estimateList)
    plt.title('{} estimate: {:.4f}'.format(name, slope_estimate))
    plt.tight_layout()

    plt.subplot(212)
    plt.plot(x, confidenceList)
    plt.title('confidence')
    plt.tight_layout()

    plt.show()

    return slope_estimate

if __name__ == '__main__':
    # load inputs
    mag = sio.loadmat('stft/case1_mag_20ms.mat')['mag']  # nCH, F, T
    phs = sio.loadmat('stft/case1_phs_20ms.mat')['phs']  # nCH, F, T
    mag = torch.FloatTensor(mag)
    phs = torch.FloatTensor(phs)
    # reshape into (N, F, nCH, T)
    mag = mag.permute(1, 0, 2).unsqueeze(0)
    phs = phs.permute(1, 0, 2).unsqueeze(0)

    pmd = intermic_mag_diff(mag)
    ppd = intermic_phs_diff(phs)

    pmd = pmd.numpy()
    ppd = ppd.numpy()

    pair_id = 2  # 0..2
    _, F, _, T = np.shape(ppd)
    t_relative = 0.35
    t = math.ceil(T*t_relative)

    plt.close('all')

    # plt.figure()
    # plt.plot(pmd[0, :, pair_id, t])
    # plt.title('PMD at {}T'.format(t_relative))

    plt.figure()
    plt.plot(ppd[0, :, pair_id, t])
    plt.title('PPD at {}T'.format(t_relative))

    plt.show()

    # ppd estimation
    diff_ppd = np.diff(ppd, axis=1)
    slope_estimate = estimate_value(diff_ppd, pair_id, 'slope')
    # pmd_estimate = estimate_value(pmd, pair_id, 'pmd')
    pmd_estimate = pmd
    tau_difference_estimate = slope_estimate/(2*math.pi*8000/160)
    print('tau difference estimate: {:.4g}s, pmd estimate: {:.4g}'.
          format(tau_difference_estimate, pmd_estimate[0, pair_id, t]))
    # implement pmd estimation
