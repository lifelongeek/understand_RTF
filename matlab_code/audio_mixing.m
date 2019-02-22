clear all; clc; close all;

nMic = 4;
nSrc = 2;
mic_xyz = zeros(nMic, 3);
src_xyz = zeros(nSrc, 3);

% Room
room = [5, 5, 3];
center = [room(1)/2, room(2)/2, room(3)/2];


% Mic
mic_z = 1;
mic_theta_list = 2*pi/16*[7.5, 8, 8.5, 9];
mic_r_list = [1.5, 1.5, 1.5, 1.5];
mic_xyz(:, 3) = mic_z;
for m=1:nMic
    mic_xyz(m,1) = center(1) + mic_r_list(m)*cos(mic_theta_list(m));
    mic_xyz(m,2) = center(2) + mic_r_list(m)*sin(mic_theta_list(m));
end

% Src (as far as possible)
src_z_list = [1.6, 1.8];
src_theta_list = 2*pi/16*[13, 2];
src_r_list = [2.0, 2.5];

for s=1:nSrc
    src_xyz(s,1) = center(1) + src_r_list(s)*cos(src_theta_list(s));
    src_xyz(s,2) = center(2) + src_r_list(s)*sin(src_theta_list(s));
    src_xyz(s,3) = src_z_list(s);
end

% calculate distance
distance = zeros(nSrc, nMic);
for s=1:nSrc
    for m=1:nMic
        distance(s, m) = sqrt(sum((mic_xyz(m, :) - src_xyz(s, :)).^2));
    end
end
disp('distance between mic & src = ');
disp(distance);

mean_dist = mean(distance, 2);
std_dist = std(distance, 0, 2);


% visualize spatial distribution
figure;
scatter3(mic_xyz(:, 1), mic_xyz(:, 2), mic_xyz(:, 3), 'filled');
hold on;
scatter3(src_xyz(1, 1), src_xyz(1, 2), src_xyz(1, 3), 'filled');
scatter3(src_xyz(2, 1), src_xyz(2, 2), src_xyz(2, 3), 'filled');
scatter3(center(1), center(2), center(3), 'x');
axis([0, room(1), 0, room(2), 0, room(3)]);

   

mylg = legend('mic', ['src (d = ' num2str(mean_dist(1)) 'm)'], ['src (d = ' num2str(mean_dist(2)) 'm)']);
mylg.FontSize = 14;
mylg.Location = 'northeast';
xlabel('X', 'FontSize', 14); 
ylabel('Y', 'FontSize', 14); 
zlabel('Z', 'FontSize', 14); 

hold off;

% Read audio
[x1, fs] = audioread('../audio/237-134493-0002.wav');
[x2, fs] = audioread('../audio/1320-122617-0001.wav');

c = 340;
nSample = 6400;

% Case 1. #Src =1, #Mic=4, RT60 = 0
disp('make audio for case 1');
RT60 = 0;
h11 = rir_generator(c, fs, mic_xyz(1,:), src_xyz(1, :), room, RT60, nSample);
h12 = rir_generator(c, fs, mic_xyz(2,:), src_xyz(1, :), room, RT60, nSample);
h13 = rir_generator(c, fs, mic_xyz(3,:), src_xyz(1, :), room, RT60, nSample);
h14 = rir_generator(c, fs, mic_xyz(4,:), src_xyz(1, :), room, RT60, nSample);

figure(2); 
subplot(411); plot(h11(1:200), 'LineWidth', 1.5); title('h11', 'FontSize', 14);
subplot(412); plot(h12(1:200), 'LineWidth', 1.5); title('h12', 'FontSize', 14);
subplot(413); plot(h13(1:200), 'LineWidth', 1.5); title('h13', 'FontSize', 14);
subplot(414); plot(h14(1:200), 'LineWidth', 1.5); title('h14', 'FontSize', 14);

save('RIR_s1.mat', 'h11', 'h12', 'h13', 'h14');

y11 = fftfilt(h11, x1);
y12 = fftfilt(h12, x1);
y13 = fftfilt(h13, x1);
y14 = fftfilt(h14, x1);

audiowrite('../audio/case1_y11.wav', y11, fs);
audiowrite('../audio/case1_y12.wav', y12, fs);
audiowrite('../audio/case1_y13.wav', y13, fs);
audiowrite('../audio/case1_y14.wav', y14, fs);

figure(52);
subplot(421); plot(h11(1:200), 'LineWidth', 1.5); title('h11', 'FontSize', 14);
subplot(423); plot(h12(1:200), 'LineWidth', 1.5); title('h12', 'FontSize', 14);
subplot(425); plot(h13(1:200), 'LineWidth', 1.5); title('h13', 'FontSize', 14);
subplot(427); plot(h14(1:200), 'LineWidth', 1.5); title('h14', 'FontSize', 14);

T11 = length(y11);
T12 = length(y12);
T13 = length(y13);
T14 = length(y14);

subplot(422); plot(y11(1:200), 'LineWidth', 1.5); title(['y11 (T=' num2str(T11) ')'], 'FontSize', 14);
subplot(424); plot(y12(1:200), 'LineWidth', 1.5); title(['y12 (T=' num2str(T12) ')'], 'FontSize', 14);
subplot(426); plot(y13(1:200), 'LineWidth', 1.5); title(['y13 (T=' num2str(T13) ')'], 'FontSize', 14);
subplot(428); plot(y14(1:200), 'LineWidth', 1.5); title(['y14 (T=' num2str(T14) ')'], 'FontSize', 14);


% Case 3. #Src =, #Mic=4, RT60 = 400ms
disp('make audio for case 3');
h21 = rir_generator(c, fs, mic_xyz(1,:), src_xyz(2, :), room, RT60, nSample);
h22 = rir_generator(c, fs, mic_xyz(2,:), src_xyz(2, :), room, RT60, nSample);
h23 = rir_generator(c, fs, mic_xyz(3,:), src_xyz(2, :), room, RT60, nSample);
h24 = rir_generator(c, fs, mic_xyz(4,:), src_xyz(2, :), room, RT60, nSample);

figure(3); 
subplot(411); plot(h21(1:200)); title('h21');
subplot(412); plot(h22(1:200)); title('h22');
subplot(413); plot(h23(1:200)); title('h23');
subplot(414); plot(h24(1:200)); title('h24');

save('RIR_s2.mat', 'h21', 'h22', 'h23', 'h24');

y21 = fftfilt(h21, x2);
y22 = fftfilt(h22, x2);
y23 = fftfilt(h23, x2);
y24 = fftfilt(h24, x2);

audiowrite('../audio/case3_y21.wav', y21, fs);
audiowrite('../audio/case3_y22.wav', y22, fs);
audiowrite('../audio/case3_y23.wav', y23, fs);
audiowrite('../audio/case3_y24.wav', y24, fs);

y1 = y11(1:min(length(y11), length(y21))) + y21(1:min(length(y11), length(y21)));
y2 = y12(1:min(length(y12), length(y22))) + y22(1:min(length(y12), length(y22)));
y3 = y13(1:min(length(y13), length(y23))) + y23(1:min(length(y13), length(y23)));
y4 = y14(1:min(length(y14), length(y24))) + y24(1:min(length(y14), length(y24)));

audiowrite('../audio/case3_y1.wav', y1, fs);
audiowrite('../audio/case3_y2.wav', y2, fs);
audiowrite('../audio/case3_y3.wav', y3, fs);
audiowrite('../audio/case3_y4.wav', y4, fs);




