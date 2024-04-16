function [invT] = identifyinvt(filename)
%% Loading ECG signal file
    startN =50000;
    [signal,Fs,tm]=rdsamp(filename,[],startN);
    ecgsig = signal(:,1);
    
%% Filtering
% decomposing the ECG to level 5 using sym4 wavelet
wt = modwt(ecgsig,5);
% reconstruct the ECG to covers the passband (for scale 5,[5.625,11.25]Hz)
wtrec = zeros(size(wt));
wtrec(4:4,:) = wt(4:4,:);
filt = imodwt(wtrec,'sym4');

baseline = mean(filt);
filt = filt-baseline;

%% R-peak detection
% first stage
    passmark1 = 0.1*max(ecgsig);
    [peaks1,locs1] = findpeaks(filt,tm,'MinPeakHeight',passmark1);
    passmark2 = 0.45*mean(peaks1);

% second stage
    [rpeaks,rlocs,rwidth] = findpeaks(filt,tm,'MinPeakHeight',passmark2,'MinPeakDistance',0.01);
    RRintervals = diff(rlocs);
    meanRR = mean(RRintervals);
    rIndices = [];
% matching to original peaks
    detectedRpeak=[];
    for i=1:1:length(rlocs)
        rIndices = [rIndices find(tm==rlocs(i))];
        detectedRpeak = [detectedRpeak ecgsig(find(tm==rlocs(i)))];
    end
    meanRRindices = mean(diff(rIndices));
    
%% Peaks detection
    qpeaks=[];
    qlocs=[];

    speaks=[];
    slocs=[];

    tpeaks=[];
    tlocs=[];
    durationT=[];

    ppeaks=[];
    plocs=[];
    durationP=[];

    qrslength=[];
    abnormalqrs=[];
    abnormalQRSlocs=[];
    abnormalQRSpeaks=[];
    abnormalQRSindex = [];

    prInterval = [];
    qtInterval=[];
    invertedfilt = -filt;
    for i=1:length(rpeaks)-1
        qIndex = find(tm==rlocs(i))-30;
        rIndex = find(tm==rlocs(i));
        sIndex = find(tm==rlocs(i))+30;
        pIndex = find(tm==rlocs(i))-60;

        if pIndex>0 && sIndex<length(ecgsig)
            [qp,ql,qw] = findpeaks(invertedfilt(qIndex:rIndex),tm(qIndex:rIndex),'MinPeakHeight',detectedRpeak(i)*0.1,'MinPeakDistance',0.01);
            if ql
                actualQindex = find(ql==max(ql));
                qpeaks=[qpeaks -qp(actualQindex)];
                qlocs=[qlocs ql(actualQindex)];
                pStopIndex = find(tm==ql(actualQindex));
                pStartIndex = pStopIndex-30;
            else
                pStartIndex =rIndex-30;
                pStopIndex = pStartIndex+30;
            end
            [pp,pl,pw] = findpeaks(filt(pStartIndex:pStopIndex),tm(pStartIndex:pStopIndex),'MinPeakHeight',detectedRpeak(i)*0.05,'MinPeakDistance',0);
            if pp
                actualPindex = find(pp==max(pp));
                ppeaks=[ppeaks pp(actualPindex)];
                plocs=[plocs pl(actualPindex)];
                prInterval=[prInterval rlocs(i)-(pl(actualPindex)-(pw(actualPindex)/2))];
                durationP=[durationP pw(actualPindex)];
            end
            [sp,sl,sw] = findpeaks(invertedfilt(rIndex:sIndex),tm(rIndex:sIndex),'MinPeakHeight',detectedRpeak(i)*0.1,'MinPeakDistance',0.01);
            if sp
                actualSindex = find(sl==min(sl));
                speaks=[speaks -sp(actualSindex)];
                slocs=[slocs sl(actualSindex)];
                tStartIndex=find(tm==sl(actualSindex))+10;
                %find(tm==rlocs(i))+round(0.5*(find(tm==rlocs(i+1))-find(tm==rlocs(i))));
            else
                tStartIndex =rIndex+30;
            end
            tStopIndex = tStartIndex + round(0.5*meanRRindices);
            %tStartIndex
            %tStopIndex
            %detectedRpeak(i)*0.01
            if tStopIndex<length(ecgsig)
                [tp,tl,tw] = findpeaks(abs(invertedfilt(tStartIndex:tStopIndex)),tm(tStartIndex:tStopIndex),'MinPeakHeight',detectedRpeak(i)*0.01,'MinPeakDistance',0.05);
                if tp
                    actualTindex = find(tl==min(tl));
                    tpeaks=[tpeaks filt(find(tm==tl(actualTindex)))];
                    tlocs=[tlocs tl(actualTindex)];
                    durationT=[durationT tw(actualTindex)];
                end
            end
        end
    end
    
    if tpeaks<0
        invT = true;
        disp(tlocs(find(tpeaks<0)))
    else
        invT = false;
    end
end