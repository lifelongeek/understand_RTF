import math
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import stats

def intermic_mag_diff(stft_magn, plot_enable=False):
    # input : torch tensor of shape (num batches, F, nCH, T) for stft magnitude
    # outputs :
    #           torch tensor for log(Mi/M0) (or IMD), for i = ids of mics other than 0th
    #           torch tensor for confidence in each IMD entry, same shape as IMD
    eps = 1e-8
    mag_thr = 1e-6
    minval = math.log(eps)
    maxval = 10000

    N, F, nCH, T = stft_magn.size()
    nCombination = nCH-1  # assume that reference mic appear at the first of dim

    # output = torch.FloatTensor(N, F, nCombination, T).zero_()
    imd_estimates = torch.FloatTensor(N, nCombination, T).zero_()
    imd_confidences = torch.zeros_like(imd_estimates)

    ## mask can be used if needed - to compute mean for only selected entries
    # mask = (stft_magn[:, :, 0, :] > mag_thr)  # NxFxT
    # mask = mask.float()
    # unmasked_freq_count = torch.sum(mask, dim=1)  # NxT

    imd_nomask_allPairs = torch.zeros(N, F, nCombination, T)

    for i in range(nCH-1):
        imd_nomask_allPairs[:,:,i,:] = torch.log(stft_magn[:, :, i+1, :]+eps) - \
                                       torch.log(stft_magn[:, :, 0, :] + eps)
        # imd_masked = mask*stft_magn[:, :, i+1, :]/(stft_magn[:, :, 0, :] + eps)  # NxFxT
        # imd_masked_mean = torch.sum(imd_masked, dim=1)/unmasked_freq_count  # NxT
        # imd_masked_var = torch.sum(mask*(imd_masked-imd_masked_mean.expand_as(imd_masked))**2, dim=1)/unmasked_freq_count
        # imd_confidences[:, i, :] = 1/imd_masked_var  # NxT
        imd_nomask_mean = torch.mean(imd_nomask_allPairs[:,:,i,:], dim=1)  # NxT
        imd_nomask_std = torch.std(imd_nomask_allPairs[:,:,i,:], dim=1)  # N x nComb x T
        imd_confidences[:, i, :] = 1/imd_nomask_std

        imd_estimates[:, i, :] = imd_nomask_mean

    imd_estimates = torch.clamp(imd_estimates, min=minval, max=maxval)

    if plot_enable:
        # make some plots
        imd_nomask_allPairs = imd_nomask_allPairs.numpy()
        plt.close('all')
        plt.figure()
        plt.subplot(221)
        plt.plot(imd_nomask_allPairs[0, :, 0, math.ceil(0.3 * T)])
        plt.title('IMD at 0.3T')
        plt.tight_layout()

        plt.subplot(222)
        plt.plot(imd_nomask_allPairs[0, :, 0, math.ceil(0.6 * T)])
        plt.title('IMD at 0.6T')
        plt.tight_layout()

        plt.subplot(223)
        plt.plot(stft_magn[0, :, 0, math.ceil(0.3 * T)].numpy())
        plt.title('M reference at 0.3T')
        plt.tight_layout()

        plt.subplot(224)
        plt.plot(stft_magn[0, :, 0, math.ceil(0.6 * T)].numpy())
        plt.title('M reference at 0.6T')
        plt.tight_layout()
        plt.savefig('figures/IMD.png')
        plt.show()


    # return output
    return imd_estimates, imd_confidences  # N x nCombination x T


def intermic_phs_diff(stft_phs, plot_enable=False):
    # input: torch tensor of shape (num batches, F, nCH, T) for stft phase
    # output: torch tensor for unwrapped phase differences, shape: (N, F, nCombination, T)
    PI = math.pi
    TWOPI = PI*2
    N, F, nCH, T = stft_phs.size()
    nCombination = nCH-1 # assume that reference mic appear at the first of dim

    wrapped_ipd = torch.FloatTensor(N, F, nCombination, T).zero_()

    for i in range(nCH-1):
        wrapped_ipd[:, :, i, :] = stft_phs[:, :, i+1, :] - stft_phs[:, :, 0, :]

    # wrap
    wrapped_ipd = torch.remainder(wrapped_ipd + PI, TWOPI) - PI

    # unwrap
    wrapped_ipd = wrapped_ipd.numpy()
    unwrapped_ipd = np.unwrap(wrapped_ipd, axis=1)

    if plot_enable:
        # make some plots
        plt.close('all')
        plt.figure()
        plt.subplot(221)
        plt.plot(wrapped_ipd[0, :, 0, math.ceil(0.3*T)])
        plt.title('wrapped ipd at 0.3T')
        plt.tight_layout()

        plt.subplot(223)
        plt.plot(unwrapped_ipd[0,:,0,math.ceil(0.3*T)])
        plt.title('unwrapped ipd at 0.3T')
        plt.tight_layout()

        plt.subplot(222)
        plt.plot(wrapped_ipd[0, :, 0, math.ceil(0.6*T)])
        plt.title('wrapped ipd at 0.6T')
        plt.tight_layout()

        plt.subplot(224)
        plt.plot(unwrapped_ipd[0,:,0,math.ceil(0.6*T)])
        plt.title('unwrapped ipd at 0.6T')
        plt.tight_layout()
        plt.savefig('figures/IPD.png')
        plt.show()

        plt.close('all')
        plt.figure()
        plt.subplot(211)
        plt.plot(np.diff(wrapped_ipd[0, :, 0, math.ceil(0.3 * T)]))
        plt.title('diff(ipd) at 0.3T')
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(np.diff(unwrapped_ipd[0, :, 0, math.ceil(0.6 * T)]))
        plt.title('diff(ipd) at 0.6T')
        plt.tight_layout()
        plt.savefig('figures/diff(IPD).png')
        plt.show()

    unwrapped_ipd = torch.FloatTensor(unwrapped_ipd)
    return unwrapped_ipd  # (N, F, nCombination, T)


def intermic_tau_diff(stft_phs, plot_enable=False):
    # input: torch tensor of shape (num batches, F, nCH, T) for stft phase
    # output: torch tensor for tau differences, and corresponding confidences
    ipd = intermic_phs_diff(stft_phs, plot_enable=plot_enable)  # N, F, nComb, T
    ipd = ipd.numpy()
    diff_ipd = np.diff(ipd, axis=1)  # N, F, nComb, T
    slope_estimates = np.mean(diff_ipd, 1)  # N, nComb, T
    tau_diff_estimates = slope_estimates/(2*math.pi*8000/160)   # N, nComb, T
    tau_diff_confidences = 1/(np.std(diff_ipd, 1) + 1e-8)   # N, nComb, T

    if plot_enable:
        # plot ipd at highest confidence, for 1 pair
        plt.close('all')
        plt.figure()
        plt.subplot(211)
        plt.plot(ipd[0,:,0, np.argmax(tau_diff_confidences[0, 0, :])])
        plt.title('IPD at argmax(Confidence(t))')

        plt.subplot(212)
        plt.plot(np.diff(ipd[0,:,0, np.argmax(tau_diff_confidences[0, 0, :])]))
        plt.title('diff(IPD) at argmax(Confidence(t))')

        plt.savefig('figures/IPD(argmax(Confidence(t)).png')
        plt.show()

    return tau_diff_estimates, tau_diff_confidences  # (num batches, nCombination, T)


if __name__ == '__main__':
    pair_id = 0  # 0..2
    t_relative = 0.35

    # load inputs
    # mix_id = '1919-142785-0009'  # bad case
    mix_id = 'case1'    # good case
    mag = sio.loadmat('stft/{}_mag_20ms.mat'.format(mix_id))['mag']  # nCH, F, T
    phs = sio.loadmat('stft/{}_phs_20ms.mat'.format(mix_id))['phs']  # nCH, F, T
    mag = torch.FloatTensor(mag)
    phs = torch.FloatTensor(phs)
    # reshape into (N, F, nCH, T)
    mag = mag.permute(1, 0, 2).unsqueeze(0)
    phs = phs.permute(1, 0, 2).unsqueeze(0)

    imd_estimates, imd_confidences = intermic_mag_diff(mag, plot_enable=True)
    tau_diff_estimates, tau_diff_confidences = intermic_tau_diff(phs, plot_enable=True)

    imd_estimates = imd_estimates.numpy()
    imd_confidences = imd_confidences.numpy()

    _, F, _, T = np.shape(mag)

    plt.close('all')
    # tau difference errors and confidences
    plt.figure()
    plt.subplot(211)
    plt.plot(tau_diff_estimates[0, pair_id, :])
    plt.title('tau difference for pair {}-1'.format(pair_id + 2))
    plt.ylabel('delay difference [s]')
    plt.tight_layout()
    plt.subplot(212)
    plt.plot(tau_diff_confidences[0, pair_id, :])
    plt.title('tau confidence for pair {}-1'.format(pair_id + 2))
    plt.xlabel('time frame')
    plt.tight_layout()
    plt.savefig('figures/TauDiffVsTime.png')
    plt.show()

    # IMD errors and confidences
    plt.close('all')
    plt.figure()
    plt.subplot(211)
    plt.plot(imd_estimates[0, pair_id, :])
    plt.title('IMD for pair {}-1'.format(pair_id+2))
    plt.tight_layout()
    plt.subplot(212)
    plt.plot(imd_confidences[0, pair_id, :])
    plt.title('IMD confidence for pair {}-1'.format(pair_id+2))
    plt.xlabel('time frame')
    plt.tight_layout()

    plt.savefig('figures/IMDvsTime.png')
    plt.show()

    # estimates for all pairs
    nPairs = np.shape(tau_diff_estimates)[1]
    tau_differences = [tau_diff_estimates[0, p, np.argmax(tau_diff_confidences[0, p])]
                       for p in range(nPairs)]
    imd_means = np.mean(imd_estimates[0, :, :], axis=1)
    print('estimated tau differences: {},\n' 
          'estimated imds: {}'.format(tau_differences, imd_means))
