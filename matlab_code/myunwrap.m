function unwrapped = myunwrap(phaseMod, args)
if nargin == 0
    load('../stft/case1_ppd_20ms_wrap_mod.mat', 'ppd_wrap_mod');
    load('../stft/case1_mag_20ms.mat', 'mag');
    %load('../stft/case3_mix_ppd_20ms_wrap_mod.mat', 'ppd_wrap_mod');
    %load('../stft/case3_mix_mag_20ms.mat', 'mag');
    args.time = ceil(size(ppd_wrap_mod, 3)*0.35);
    args.pair = [1 2];
    args.pairID = 1;
    phaseMod = ppd_wrap_mod(args.pairID,:,args.time);
    args.tol = pi;
    args.meth = 'matlab';
    args.mThr = 3e-4;
    args.magn = mag;
    
    toPlot = true;
else 
    toPlot = false;
end

if strcmp(args.meth, 'matlab')
    unwrapped = unwrap(phaseMod, args.tol);
elseif strcmp(args.meth, 'replica')
    unwrapped = zeros(size(phaseMod));
    F = size(phaseMod, 2);
    %from https://www.mathworks.com/help/dsp/ref/unwrap.html:
    k = 0;
    for f=1:F-1
        unwrapped(f) = phaseMod(f)+2*pi*k;
        if abs(phaseMod(f+1)-phaseMod(f)) > abs(args.tol)
            if phaseMod(f+1)<phaseMod(f)
                k = k+1;
            else
                k = k-1;
            end
        end
    end
    unwrapped(f+1) = phaseMod(f+1)+2*pi*k;
    unwrapped(1,:) = unwrapped;

elseif strcmp(args.meth, 'NoiseRobust')
    unwrapped = zeros(size(phaseMod));
    F = size(phaseMod, 2);
    % from 'Noise robust linear dynamic system for phase unwrapping and
    % smoothing'
    tau = args.tau;
    accum = 0;
    for f=1:F
        accum = accum + ((1-tau)^(f-1))*phaseMod(f);
        unwrapped(f) = accum;
    end
    unwrapped = unwrapped * tau;
    
elseif strcmp(args.meth, 'PhaseDenoise')
    phaseDenoised = lowpass(phaseMod(1,:,:), 4000, 16000);
    unwrapped = unwrap(phaseDenoised, args.tol);
elseif strcmp(args.meth, 'LowPassFilt')
    F = size(phaseMod, 2);
    unwrapped = unwrap(phaseMod(1,1:ceil(F/2),:), args.tol);
elseif strcmp(args.meth, 'RunningSlope')
    unwrapped = zeros(size(phaseMod));
    F = size(phaseMod, 2);
    runSlope = 0; % will track running mean of slope
    
    magn_frameA = args.magn(args.pair(1),:,args.time);
    magn_frameB = args.magn(args.pair(2),:,args.time);
    k = 0;
    for f=1:F-1
        if (magn_frameA(f) > args.mThr) && (magn_frameB(f)>args.mThr)
            unwrapped(f) = phaseMod(f)+2*pi*k;
            if ((abs(phaseMod(f+1)-phaseMod(f)) > abs(args.tol)) &&... 
                (magn_frameA(f+1) > args.mThr) && (magn_frameB(f+1)>args.mThr))
                if phaseMod(f+1)<phaseMod(f)
                    k = k+1;
                else
                    k = k-1;
                end
            end
        else
            if f>1
                unwrapped(f) = unwrapped(f-1) + runSlope;
            else
                unwrapped(1) = 0;
            end    
        end

        %update running mean
        if f>1
            runSlope = 0.9*runSlope + 0.1*(unwrapped(f)-unwrapped(f-1));
        end
    end
    if magn_frameA(f+1)>args.mThr && magn_frameB(f+1)>args.mThr
        unwrapped(f+1) = phaseMod(f+1)+2*pi*k;
    else
        unwrapped(f+1) = unwrapped(f)+runSlope;
    end

else
    error('method "%s" is not supported', args.meth);
end

if toPlot == true
    T = size(ppd_wrap_mod, 3);
    F = size(ppd_wrap_mod, 2);
    
    f1 = figure(1);
    subplot(311); plot(phaseMod(1,:)); title('ppd');
    subplot(312); plot(unwrapped(1,:)); title('unwrapped ppd');
    diffUnwrapped = diff(unwrapped);
    subplot(313); 
    binWidth = 0.05;
    nbins = ceil((max(diffUnwrapped)-min(diffUnwrapped))/binWidth);
    histogram(diffUnwrapped, nbins); title('slope distribution');
    meanDiffUnwrapped = mean(diffUnwrapped);
    
    figure(2);
    subplot(411); plot(phaseMod); title('ppd');
    subplot(412); plot(unwrapped(1,:)); title('unwrapped ppd');
    magn_frameA = args.magn(args.pair(1),:,args.time);
    magn_frameB = args.magn(args.pair(2),:,args.time);
    subplot(413); plot(magn_frameA); title(sprintf('fft magnitude at mic %d', args.pair(1)));
    subplot(414); plot(magn_frameB); title(sprintf('fft magnitude of mic %d', args.pair(2)));
end
end