function singleSrcMixing(src_path, mic_dir, h_dir)
if nargin==0
    src_path = '../audio/source/84-121123-0000.wav';
    mic_dir = '../audio/mic';
    h_dir = '../H';
end
[~, src_fname, ~] = fileparts(src_path);
[status, msg] = mkdir(mic_dir); % status and msg are needed for not seeing warnings if folder already exists
assert(status, strcat('something went wrong when creating a folder. Error message: ', msg))
[status, msg] = mkdir(h_dir);
assert(status, strcat('something went wrong when creating a folder. Error message: ', msg))

room = [5 5 3];
nMic = 2;
minSrcMicDist = 1;
while 1
    % put Src at a random position
    mic_xyz_room = room.*rand(nMic, 3);
    src_xyz_room = room.*rand(1, 3);
    % accept only if distance between source and closest mic is greater
    % than minDist
    distances = sqrt(sum((mic_xyz_room-src_xyz_room).^2, 2));
    if min(distances) > minSrcMicDist  
        break;
    end
end

% %% visualize positions
% figure;
% scatter3(mic_xyz_room(:, 1), mic_xyz_room(:, 2), mic_xyz_room(:, 3), 'filled');
% hold on;
% scatter3(src_xyz_room(1, 1), src_xyz_room(1, 2), src_xyz_room(1, 3), 'filled');
% % scatter3(center(1), center(2), center(3), 'x');
% axis([0, room(1), 0, room(2), 0, room(3)]);

%% mixing
% Read audio
[s1, fs] = audioread(src_path);
c = 340;
n_h_sample = 1024;  % more than enough for no reverberation

RT60=0;

h11 = rir_generator(c, fs, mic_xyz_room(1,:), src_xyz_room(1, :), room, RT60, n_h_sample);
h12 = rir_generator(c, fs, mic_xyz_room(2,:), src_xyz_room(1, :), room, RT60, n_h_sample);

% figure(2); 
% subplot(211); plot(h11(1:400), 'LineWidth', 1.5); title('h11', 'FontSize', 14);
% subplot(212); plot(h12(1:400), 'LineWidth', 1.5); title('h12', 'FontSize', 14);
save(fullfile(h_dir, sprintf('%s.mat', src_fname)), 'h11', 'h12');

m11 = fftfilt(h11, s1);
m12 = fftfilt(h12, s1);

audiowrite(fullfile(mic_dir, sprintf('%s_m1.wav', src_fname)), m11, fs);
audiowrite(fullfile(mic_dir, sprintf('%s_m2.wav', src_fname)), m12, fs);
end