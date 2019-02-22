from intermic_diff import intermic_tau_diff_mean, intermic_tau_diff_distribution_maxima, intermic_mag_diff
import os
import numpy as np, scipy.io as sio, torch, matplotlib.pyplot as plt
import argparse


def relative_err(estimate, truth):
    err = np.abs(estimate/(truth+1e-10) - 1)
    return err


def absolute_err(estimate, truth):
    err = np.abs(estimate - truth)
    return err


if __name__ == '__main__':
    src_dir = 'audio/source'
    h_dir = 'H'
    stft_dir = 'stft'
    main_plot_enable = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--window_duration', type=int, default=20)
    parser.add_argument('--method', type=str, default='mean', help="'mean' or 'distribution_maxima'")
    parser.add_argument('--top_percent', type=float, default=50, help='percent of top bins to compute mean over')
    args = parser.parse_args()

    imd_errors, tau_diff_errors = [], []
    tau_diff_estimate_list, tau_diff_truth_list = [], []
    imd_estimate_list, imd_truth_list = [], []
    # for each wav file in src_dir:
    for filename in os.listdir(src_dir):
        if filename.endswith(".wav"):
            mix_id = filename[:-4]

            mag = sio.loadmat(os.path.join(stft_dir, '{}_mag_{}ms.mat'.format(mix_id, args.window_duration)))['mag']
            phs = sio.loadmat(os.path.join(stft_dir, '{}_phs_{}ms.mat'.format(mix_id, args.window_duration)))['phs']
            mag = torch.FloatTensor(mag)
            phs = torch.FloatTensor(phs)
            # reshape into (N, F, nCH, T)
            mag = mag.permute(1, 0, 2).unsqueeze(0)
            phs = phs.permute(1, 0, 2).unsqueeze(0)

            # # uncomment this block if imd is needed
            # imd_estimates, imd_confidences = intermic_mag_diff(mag)
            # imd_estimates = imd_estimates.numpy()
            # imd_confidences = imd_confidences.numpy()
            # imd_estimate = np.asscalar(np.mean(imd_estimates[0, :, :], axis=1))
            # # imd_estimate = imd_estimates[0, 0, np.argmax(imd_confidences[0, 0, :])]
            # imd_estimate_list.append(imd_estimate)

            # choosing desired method
            if args.method == 'mean':
                # estimate by taking mean:
                tau_diff_estimates, tau_diff_confidences = intermic_tau_diff_mean(phs)
                tau_diff_estimate = -tau_diff_estimates[0, 0, np.argmax(tau_diff_confidences[0,0,:])]
            elif args.method == 'distribution_maxima':
                # estimate by taking mean of small set of tallest bins
                tau_diff_estimate = - intermic_tau_diff_distribution_maxima(phs, percent_top=args.top_percent, binWidth=0.0001)
                tau_diff_estimate = np.asscalar(tau_diff_estimate)
            else:
                assert False, "method is not supported"

            tau_diff_estimate_list.append(tau_diff_estimate)

            # find true values
            h_dict = sio.loadmat(os.path.join(h_dir, '{}.mat'.format(mix_id)))
            h11 = h_dict['h11']
            h12 = h_dict['h12']

            imd_true = np.log(np.max(h12)/np.max(h11))
            imd_truth_list.append(imd_true)
            tau_diff_true = (np.argmax(h12)-np.argmax(h11))/16000
            tau_diff_truth_list.append(tau_diff_true)

            # # uncomment these lines if imd is needed
            # imd_err = absolute_err(imd_estimate, imd_true)
            # imd_errors.append(imd_err)

            tau_diff_err = absolute_err(tau_diff_estimate, tau_diff_true)
            tau_diff_errors.append(tau_diff_err)

            # print('mix_id: {}'.format(mix_id))
            # print('tau_diff_estimate: {:.5g}s, tau_diff_true: {:.5g}s, tau_diff_error: {:.5g}s'.
            #       format(tau_diff_estimate, tau_diff_true, tau_diff_err))
            # print('imd_estimate: {:.5g}, imd_true: {:.5g}, imd_error: {:.5g}\n'.
            #       format(imd_estimate, imd_true, imd_err))
    #
    # print('avg abs tau diff error: {:.5g}, avg abs imd error: {:.5g}'.
    #       format(np.mean(tau_diff_errors), np.mean(imd_errors)))
    if args.method=='mean':
        print('window: {}ms, method: "mean", avg error: {:.4g}ms'.
              format(args.window_duration, np.mean(tau_diff_errors)*1000))
    else:
        print('window: {}ms, method: "distribution_maxima", top_percent: {}, avg_error: {:.4g}ms'.
              format(args.window_duration, args.top_percent, np.mean(tau_diff_errors)*1000))

    if main_plot_enable:
        plt.close('all')
        # # plotting errors
        # plt.figure()
        # plt.subplot(212)
        # plt.plot(imd_errors)
        # plt.plot(np.abs(imd_truth_list))
        # plt.legend(['|errors|', '|true IMDs|'])
        # plt.title('IMD errors')
        # plt.xlabel('sample id')
        # plt.ylabel('absolute error')
        # plt.tight_layout()
        #
        # plt.subplot(211)
        # plt.plot(tau_diff_errors)
        # plt.plot(np.abs(tau_diff_truth_list))
        # plt.legend(['|errors|', '|true tau diffs|'])
        # plt.title('tau difference errors')
        # plt.xlabel('sample id')
        # plt.ylabel('time [s]')
        # plt.tight_layout()
        # plt.savefig('figures/errors_for_all.png')

        # juxtaposing estimates with truths
        plt.figure()
        # plt.subplot(211)
        plt.plot(tau_diff_estimate_list)
        plt.plot(tau_diff_truth_list)
        plt.legend(['estimates', 'ground truths'])
        plt.ylabel('tau difference [s]')
        plt.xlabel('sample id')
        plt.tight_layout()

        # plt.subplot(212)
        # plt.plot(imd_estimate_list)
        # plt.plot(imd_truth_list)
        # plt.legend(['estimates', 'ground truths'])
        # plt.ylabel('log(M1/M0)')
        # plt.tight_layout()
        plt.savefig('figures/value_comparison_for_all.png')
        plt.show()
