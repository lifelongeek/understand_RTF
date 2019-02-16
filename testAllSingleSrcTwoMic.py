from intermic_diff import intermic_tau_diff, intermic_mag_diff
import os
import numpy as np, scipy.io as sio, torch, matplotlib.pyplot as plt


def normalized_abs_loss(estimate, truth):
    loss = np.abs(estimate/(truth+1e-10) - 1)
    return loss


if __name__ == '__main__':
    src_dir = 'audio/source'
    h_dir = 'H'
    stft_dir = 'stft'

    imd_losses, tau_diff_losses = [], []
    # for each wav file in src_dir:
    for filename in os.listdir(src_dir):
        if filename.endswith(".wav"):
            mix_id = filename[:-4]

            mag = sio.loadmat(os.path.join(stft_dir, '{}_mag_20ms.mat'.format(mix_id)))['mag']
            phs = sio.loadmat(os.path.join(stft_dir, '{}_phs_20ms.mat'.format(mix_id)))['phs']
            mag = torch.FloatTensor(mag)
            phs = torch.FloatTensor(phs)
            # reshape into (N, F, nCH, T)
            mag = mag.permute(1, 0, 2).unsqueeze(0)
            phs = phs.permute(1, 0, 2).unsqueeze(0)

            imd_estimates, imd_confidences = intermic_mag_diff(mag)
            imd_estimates = imd_estimates.numpy()
            imd_confidences = imd_confidences.numpy()
            imd_estimate = np.mean(imd_estimates[0, :, :], axis=1)
            # imd_estimate = imd_estimates[0, 0, np.argmax(imd_confidences[0, 0, :])]

            tau_diff_estimates, tau_diff_confidences = intermic_tau_diff(phs)
            tau_diff_estimate = tau_diff_estimates[0, 0, np.argmax(tau_diff_confidences[0,0,:])]

            # find true values
            h_dict = sio.loadmat(os.path.join(h_dir, '{}.mat'.format(mix_id)))
            h11 = h_dict['h11']
            h12 = h_dict['h12']

            imd_true = np.log(np.max(h12)/np.max(h11))
            tau_diff_true = (np.argmax(h12)-np.argmax(h11))/16000

            imd_loss = normalized_abs_loss(imd_estimate, imd_true)
            tau_diff_loss = normalized_abs_loss(tau_diff_estimate, tau_diff_true)
            imd_losses.append(imd_loss)
            tau_diff_losses.append(tau_diff_loss)

    plt.close('all')
    plt.figure()
    plt.subplot(211)
    plt.plot(imd_losses)
    plt.title('IMD losses')
    plt.xlabel('sample id')
    plt.ylabel('normalized abs loss')
    plt.tight_layout()

    plt.subplot(212)
    plt.plot(tau_diff_losses)
    plt.title('tau difference losses')
    plt.xlabel('sample id')
    plt.ylabel('normalized abs loss')
    plt.tight_layout()
    plt.show()
