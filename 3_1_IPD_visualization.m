
clear all; clc; close all;

%% case 1) #src = 1, #mic = 4, RT60 = 0

figure(1); % t = {30%, 60%}, method = {'wrap', 'unwrap'}, micpair = {close, distant}, window = {'20ms', '500ms'}
load('stft/case1_ppd_20ms_wrap_mod.mat');

T = size(ppd_wrap_mod, 3);
F = size(ppd_wrap_mod, 2);
Flist = [1, ceil(F*0.25), ceil(F*0.5), ceil(F*0.75), F];
Flabels = {'0', '2000', '4000', '6000', '8000'};
T30 = ceil(T*0.3);
T60 = ceil(T*0.6);

h1 = subplot(441); plot(ppd_wrap_mod(1, :, T30), 'k', 'Linewidth', 2);  title('T = 30%', 'FontSize', 12); ylabel('close', 'FontSize', 14); xticks(Flist); xticklabels(Flabels); xlim([1, F]); % 30%, unwrap, 20ms, close
h2 = subplot(442); plot(ppd_wrap_mod(1, :, T60), 'k', 'Linewidth', 2);  title('T = 60%', 'FontSize', 12); xticks(Flist); xticklabels(Flabels); xlim([1, F]); % 60%, unwrap, 20ms, close
h3 = subplot(443); plot(unwrap(ppd_wrap_mod(1, :, T30)), 'k', 'Linewidth', 2);  title('T = 30%', 'FontSize', 12); xticks(Flist); xticklabels(Flabels); xlim([1, F]); % 30%, wrap, 20ms, close
h4 = subplot(444); plot(unwrap(ppd_wrap_mod(1, :, T60)), 'k', 'Linewidth', 2);  title('T = 60%', 'FontSize', 12); xticks(Flist); xticklabels(Flabels); xlim([1, F]); % 60%, wrap, 20ms, close
h5 = subplot(445); plot(ppd_wrap_mod(3, :, T30), 'k', 'Linewidth', 2);  ylabel('distant', 'FontSize', 14); xticks(Flist); xticklabels(Flabels); xlim([1, F]);% 30%, unwrap, 20ms, distant
h6 = subplot(446); plot(ppd_wrap_mod(3, :, T60), 'k', 'Linewidth', 2);  xticks(Flist); xticklabels(Flabels); xlim([1, F]); % 60%, unwrap, 20ms, distant
h7 = subplot(447); plot(unwrap(ppd_wrap_mod(3, :, T30)), 'k', 'Linewidth', 2);  xticks(Flist); xticklabels(Flabels); xlim([1, F]); % 30%, wrap, 20ms, distant
h8 = subplot(448); plot(unwrap(ppd_wrap_mod(3, :, T60)), 'k', 'Linewidth', 2);  xticks(Flist); xticklabels(Flabels); xlim([1, F]); % 60%, wrap, 20ms, distant

load('stft/case1_ppd_500ms_wrap_mod.mat');

T = size(ppd_wrap_mod, 3);
F = size(ppd_wrap_mod, 2);
Flist = [1, ceil(F*0.25), ceil(F*0.5), ceil(F*0.75), F];
T30 = ceil(T*0.3);
T60 = ceil(T*0.6);

h9 = subplot(4,4,9); plot(ppd_wrap_mod(1, :, T30), 'k', 'Linewidth', 1.5);  ylabel('close', 'FontSize', 14);  xticks(Flist); xticklabels(Flabels); xlim([1, F]);% 30%, unwrap, 500ms, close
h10 = subplot(4,4,10); plot(ppd_wrap_mod(1, :, T60), 'k', 'Linewidth', 1.5);  xticks(Flist); xticklabels(Flabels); xlim([1, F]); % 60%, unwrap, 500ms, close
h11 = subplot(4,4,11); plot(unwrap(ppd_wrap_mod(1, :, T30)), 'k', 'Linewidth', 1.5);  xticks(Flist); xticklabels(Flabels); xlim([1, F]); % 30%, wrap, 500ms, close
h12 = subplot(4,4,12); plot(unwrap(ppd_wrap_mod(1, :, T60)), 'k', 'Linewidth', 1.5);  xticks(Flist); xticklabels(Flabels); xlim([1, F]); % 60%, wrap, 500ms, close
h13 = subplot(4,4,13); plot(ppd_wrap_mod(3, :, T30), 'k', 'Linewidth', 1.5);  ylabel('distant', 'FontSize', 14); xticks(Flist); xticklabels(Flabels); xlim([1, F]); xlabel('Frequency (Hz)', 'FontSize', 12);  % 30%, unwrap, 500ms, distant
h14 = subplot(4,4,14); plot(ppd_wrap_mod(3, :, T60), 'k', 'Linewidth', 1.5);  xticks(Flist); xticklabels(Flabels); xlim([1, F]); xlabel('Frequency (Hz)', 'FontSize', 12); % 60%, unwrap, 500ms, distant
h15 = subplot(4,4,15); plot(unwrap(ppd_wrap_mod(3, :, T30)), 'k', 'Linewidth', 1.5); xticks(Flist); xticklabels(Flabels); xlim([1, F]); xlabel('Frequency (Hz)', 'FontSize', 12);  % 30%, wrap, 500ms, distant
h16 = subplot(4,4,16); plot(unwrap(ppd_wrap_mod(3, :, T60)), 'k', 'Linewidth', 1.5);  xticks(Flist); xticklabels(Flabels); xlim([1, F]);  xlabel('Frequency (Hz)', 'FontSize', 12);  % 60%, wrap, 500ms, distant

p1 = get(h1, 'position');
p2 = get(h2, 'position');
p3 = get(h3, 'position');
p4 = get(h4, 'position');
p5 = get(h5, 'position');
p9 = get(h9, 'position');
p13 = get(h13, 'position');

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.2, 0.2, 0.6, 0.7]);

%width = p1(1)+p1(3)-p2(1);
h12 = axes('position', [p2(1)*0.9 p2(2) 0 p2(4)*1.2], 'visible', 'off');
title('Pi-Pj', 'FontSize', 14, 'visible', 'on'); 

%width = p3(1)+p3(3)-p4(1);
h34 = axes('position', [p4(1)*0.96 p4(2) 0 p4(4)*1.2], 'visible', 'off');
title('U(Pi-Pj)', 'FontSize', 14, 'visible', 'on'); 

height = p1(2)+p1(4)-p5(2);
h15 = axes('position', [p5(1)*0.7 p5(2) p5(3) height], 'visible', 'off');
ylabel('20ms', 'FontSize', 14, 'visible', 'on'); 

height = p9(2)+p9(4)-p13(2);
h913 = axes('position', [p13(1)*0.7 p13(2) p13(3) height], 'visible', 'off');
ylabel('500ms', 'FontSize', 14, 'visible', 'on'); 


%% case 3) #src = 2, #mic = 4, RT60 = 0
load('stft/case1_ppd_20ms_wrap_mod.mat');
ppd_s1 = ppd_wrap_mod;
load('stft/case3_s2_ppd_20ms_wrap_mod.mat');
ppd_s2 = ppd_wrap_mod;
load('stft/case3_mix_ppd_20ms_wrap_mod.mat');
ppd_mix = ppd_wrap_mod;

T = size(ppd_mix, 3);
F = size(ppd_mix, 2);
Flist = [1, ceil(F*0.25), ceil(F*0.5), ceil(F*0.75), F];
Flabels = {'0', '2000', '4000', '6000', '8000'};
T30 = ceil(T*0.3);
T60 = ceil(T*0.6);


figure(2);  % close mic pair
subplot(321); plot(ppd_s1(1, :, T30), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); title('Pi-Pj', 'FontSize', 16); ylabel('source1', 'FontSize', 16);
subplot(322); plot(unwrap(ppd_s1(1, :, T30)), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); title('U(Pi-Pj)', 'FontSize', 16); 
subplot(323); plot(ppd_s2(1, :, T30), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); ylabel('source2', 'FontSize', 16);
subplot(324); plot(unwrap(ppd_s2(1, :, T30)), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]);
subplot(325); plot(ppd_mix(1, :, T30), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); ylabel('mixture', 'FontSize', 16);
subplot(326); plot(unwrap(ppd_mix(1, :, T30)), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]);


figure(3); % distant mic pair
subplot(321); plot(ppd_s1(3, :, T30), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); title('Pi-Pj', 'FontSize', 16); ylabel('source1', 'FontSize', 16);
subplot(322); plot(unwrap(ppd_s1(3, :, T30)), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); title('U(Pi-Pj)', 'FontSize', 16); 
subplot(323); plot(ppd_s2(3, :, T30), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); ylabel('source2', 'FontSize', 16);
subplot(324); plot(unwrap(ppd_s2(3, :, T30)), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]);
subplot(325); plot(ppd_mix(3, :, T30), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); ylabel('mixture', 'FontSize', 16);
subplot(326); plot(unwrap(ppd_mix(3, :, T30)), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]);


Ts1_s2max = 873;
Ts1_s2min = 1080;

figure(4); 
subplot(421); plot(ppd_s1(1, :, Ts1_s2max), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); title('Pi-Pj', 'FontSize', 16); ylabel('source1', 'FontSize', 14);
subplot(422); plot(unwrap(ppd_s1(1, :, Ts1_s2max)), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); title('U(Pi-Pj)', 'FontSize', 16); 
subplot(423); plot(ppd_s2(1, :, Ts1_s2min), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); ylabel('source2', 'FontSize', 14);
subplot(424); plot(unwrap(ppd_s2(1, :, Ts1_s2min)), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]);
subplot(425); plot(ppd_mix(1, :, Ts1_s2max), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); ylabel('mix (t=s1/s2 max)', 'FontSize', 14);
subplot(426); plot(unwrap(ppd_mix(1, :, Ts1_s2max)), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]);
subplot(427); plot(ppd_mix(1, :, Ts1_s2min), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]); ylabel('mix (t=s1/s2 min)', 'FontSize', 14);
subplot(428); plot(unwrap(-ppd_mix(1, :, Ts1_s2min)), 'k', 'LineWidth', 2); xticks(Flist); xticklabels(Flabels); xlim([1, F]);



