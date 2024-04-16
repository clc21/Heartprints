function meanQRSlength = calcQRSlen(filename)
%% Loading ECG signal file
    startN =50000;
    [signal,Fs,tm]=rdsamp(filename,[],startN);
    ecgsig = signal(:,1);

%% Enchancing R-peaks in ECG signal
% decomposing the ECG to level 5 using sym4 wavelet
wt = modwt(ecgsig,4);
% reconstruct the ECG to covers the passband (for scale 5,[5.625,11.25]Hz)
wtrec = zeros(size(wt));
wtrec(4:5,:) = wt(4:5,:);
filt = imodwt(wtrec,'sym4');

baseline = mean(filt);
filt = filt-baseline;


%% R-peak-finding algorithm
% squared-absolute values of signal approximation from wavelet coefficients
filt2 = filt;

% first stage
passmark1 = 0.1*max(filt2);
[peaks1,locs1] = findpeaks(filt2,tm,'MinPeakHeight',passmark1);
passmark2 = 0.45*mean(peaks1);
% second stage
%400 ms is the minimum RR-interval ever recorded
[rpeaks,rlocs] = findpeaks(filt2,tm,'MinPeakHeight',passmark2,'MinPeakDistance',0.4);

%q-peak detection
qpeaks=[];
qlocs=[];
speaks=[];
slocs=[];
tpeaks=[];
tlocs=[];
qrslength=[];
invertedfilt = -filt;

for i=1:length(rpeaks)-1
    qIndex = find(tm==rlocs(i))-30;
    rIndex = find(tm==rlocs(i));
    sIndex = find(tm==rlocs(i))+30;
    tIndex = find(tm==rlocs(i+1));
    
    [qp,ql] = findpeaks(invertedfilt(qIndex:rIndex),tm(qIndex:rIndex),'MinPeakHeight',rlocs(i)*0.25,'MinPeakDistance',0.05);
    if qp
        qpeaks=[qpeaks -qp];
        qlocs=[qlocs ql];
    end
    
    [sp,sl] = findpeaks(invertedfilt(rIndex:sIndex),tm(rIndex:sIndex),'MinPeakHeight',rlocs(i)*0.25,'MinPeakDistance',0.01);
    if sp
        speaks=[speaks -sp(1)];
        slocs=[slocs sl(1)];
    end
    
    [tp,tl] = findpeaks(abs(invertedfilt(sIndex:tIndex)),tm(sIndex:tIndex),'MinPeakHeight',rlocs(i)*0.4,'MinPeakDistance',0.01);
    if tp
        tpeaks=[tpeaks tp];
        %tpeaks=[tpeaks filt(find(tm==tl(1)))];
        tlocs=[tlocs tl];
    end
    
    if (qp&sp)
        qrslength=[qrslength sl-ql];
    end
    
end

meanQRSlength = mean(qrslength)*1000;
end



