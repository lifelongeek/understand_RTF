source_dir = 'audio/source';
mic_dir = 'audio/mic';
h_dir = 'H';

src_files = dir(fullfile(source_dir,'*.wav'));
for f = src_files'
    singleSrcMixing(fullfile(source_dir, f.name), mic_dir, h_dir);
end