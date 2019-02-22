load('../stft/case1_ppd_20ms_wrap_mod.mat', 'ppd_wrap_mod');
% load('../stft/case3_mix_ppd_20ms_wrap_mod.mat', 'ppd_wrap_mod');
% load('../stft/case1_mag_20ms.mat', 'mag');
% args.magn = mag;
T = size(ppd_wrap_mod, 3);
tList = ceil(T * (0.05:0.05:0.95));
pairID = 1;
args.meth = 'matlab'; % unwrap method
args.tol = pi;
num_top = 10; % number of tallest bins to consider when estimating slope within a frame

estimateList = zeros(size(tList));
confidenceList = zeros(size(tList));
for i = 1:size(tList, 2)
    ppd_wrapped = ppd_wrap_mod(pairID, :, tList(1, i));
    uwframe = myunwrap(ppd_wrapped, args);
    estimateList(1,i) = frame_slope_estimate(uwframe, num_top);
    confidenceList(1,i) = 1/std(diff(uwframe));
end

slope_estimate = sum(estimateList.*confidenceList)/sum(confidenceList)

figure();
x = 0.05:0.05:0.95;
subplot(211); plot(x, estimateList); title('slope estimate');
subplot(212); plot(x, confidenceList); title('confidence');
