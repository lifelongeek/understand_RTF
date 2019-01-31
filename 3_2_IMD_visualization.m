clear all; clc; close all;

%% case 1) #src = 1, #mic = 4, RT60 = 0

load('stft/case1_mag_20ms.mat');
T = size(mag, 3);
F = size(mag, 2);
Flist = [1, ceil(F*0.25), ceil(F*0.5), ceil(F*0.75), F];
Flabels = {'0', '2000', '4000', '6000', '8000'};
T30 = ceil(T*0.3);
T60 = ceil(T*0.6);

close_pmd = squeeze(log(mag(1, :, :)./mag(2, :, :)));
distant_pmd = squeeze(log(mag(1, :, :)./mag(4, :, :)));

figure(1);
subplot(211); plot(close_pmd(:, T30), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); title('log(Mi/Mj), t=30%', 'FontSize', 16);
subplot(212); plot(close_pmd(:, T60), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); title('log(Mi/Mj), t=60%', 'FontSize', 16); xlabel('Frequency(Hz)', 'FontSize', 14); 

figure(2);
subplot(211); histogram(close_pmd); title('log(Mi/Mj) (close)', 'FontSize', 16);
subplot(212); histogram(distant_pmd); title('log(Mi/Mj) (distant)', 'FontSize', 16);

%% case 3) #src = 2, #mic = 4, RT60 = 0
load('stft/case1_mag_20ms.mat');
mag_s1 = mag;
load('stft/case3_s2_mag_20ms.mat');
mag_s2 = mag;
load('stft/case3_mix_mag_20ms.mat');
mag_mix = mag;

distant_pmd_s1 = squeeze(log(mag_s1(1, :, :)./(mag_s1(4, :, :)+1e-8)));
distant_pmd_s2 = squeeze(log(mag_s2(1, :, :)./(mag_s2(4, :, :)+1e-8)));
distant_pmd_mix = squeeze(log(mag_mix(1, :, :)./(mag_mix(4, :, :)+1e-8)));


figure(3);
subplot(311); histogram(distant_pmd_s1); title('log(Mi/Mj) (s1)', 'FontSize', 16); %xlim([-5, 5]);  % [-0.57, -0.54]
subplot(312); histogram(distant_pmd_s2); title('log(Mi/Mj) (s2)', 'FontSize', 16); %xlim([-5, 5]); % [-0.57, -0.54]
subplot(313); histogram(distant_pmd_mix); title('log(Mi/Mj) (mix)', 'FontSize', 16); %xlim([-5, 5]); % [-0.57, -0.54]