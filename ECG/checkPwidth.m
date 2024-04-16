function abnormalPwave = checkPwidth(filename)
%% Loading ECG signal file
    startN =50000;
    [signal,Fs,tm]=rdsamp(filename,[],startN);
    ecgsig = signal(:,1);
    
%% Filtering
    f1=0.5;                                                                    
    f2=65;                                                                     
    Wn=[f1 f2]*2/Fs;                                                           
    N = 3;                                                                     
    [a,b] = butter(N,Wn);                                                      
    filt = filtfilt(a,b,ecgsig);
    filt = filt-mean(filt);

%% R-peak detection
% first stage
    passmark1 = 0.1*max(ecgsig);
    [peaks1,locs1] = findpeaks(filt,tm,'MinPeakHeight',passmark1);
    passmark2 = 0.45*mean(peaks1);

% second stage
    [rpeaks,rlocs,rwidth] = findpeaks(filt,tm,'MinPeakHeight',passmark2,'MinPeakDistance',0.01);
    RRintervals = diff(rlocs);
    meanRR = mean(RRintervals);
    
% matching to original peaks
    detectedRpeak=[];
    for i=1:1:length(rlocs)
        detectedRpeak = [detectedRpeak ecgsig(find(tm==rlocs(i)))];
    end
    
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
        qtime = false;
        ttime = false;

        if pIndex>0 && sIndex<length(ecgsig)
            [qp,ql,qw] = findpeaks(invertedfilt(qIndex:rIndex),tm(qIndex:rIndex),'MinPeakHeight',detectedRpeak(i)*0.1,'MinPeakDistance',0.01);
            if ql
                actualQindex = find(ql==max(ql));
                qpeaks=[qpeaks -qp(actualQindex)];
                qlocs=[qlocs ql(actualQindex)];
                pStopIndex = find(tm==ql(actualQindex));
                pStartIndex = pStopIndex-30;
                qtime = ql(actualQindex);
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
        end
    end
     if durationP>(120/1000)
         abnormalPwave = true;
     else
         abnormalPwave = false;
end