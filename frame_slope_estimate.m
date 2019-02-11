function est = frame_slope_estimate(f, num_top)
% finds mean of values in NUM_TOP bins that have highest number of elements
if nargin == 0
    load('stft/case1_ppd_20ms_wrap_mod.mat', 'ppd_wrap_mod');
    T = size(ppd_wrap_mod,3);
    f = ppd_wrap_mod(1,:,ceil(T*0.3));
    num_top=10;
end

df = diff(f);
binWidth = 0.05;
nbins = ceil((max(df)-min(df))/binWidth);
[counts, ~, bins] = histcounts(df, nbins);
[~, binIndices] = sort(counts);
binIndices = fliplr(binIndices); %indices of bins in order of decreasing bin height

binSizeList = zeros(1,num_top);
binMeanList = zeros(1,num_top);
for i = 1:num_top
    binSize = nnz(bins==binIndices(i));
    binMean = sum(df.*(bins==binIndices(i)))/binSize;
    binSizeList(1,i) = binSize;
    binMeanList(1,i) = binMean;
end
est = sum(binSizeList.*binMeanList)/sum(binSizeList);
end
