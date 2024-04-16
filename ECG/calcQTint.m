    filename = 'sddb/47';
    attr = 'ari';
%% Loading ECG signal file
    startN =75000; % for 5-minute sample
    [signal,Fs,tm]=rdsamp(filename,[],startN);
    ecgsig = 1;
    tm = tm(:,1);
    annN=rdann(filename,attr,1,startN,[],'N');
    annV=rdann(filename,attr,1,startN,[],'V');

%% Bandpass Filtering (5-18Hz)  
    fLow=5;
    fHigh=18;
    order=5;
    [b,a] = butter(order,[fLow/(Fs/2),fHigh/(Fs/2)],'bandpass');
    filtsig = filtfilt(b,a,ecgsig);
    
%% Differentiate signal
    T=1/Fs;
    h = (1/(8*T))*[-1 -2 0 2 1];
    diffSig = conv(filtsig,h,'same');

    sigSquared = diffSig.^2; % point by point squaring
    
%% Smoothing with flattop window
    winWidth1 = 60/1000;
    N1 = round(winWidth1 * Fs);
    a0 = 0.2155789;
    a1 = 0.4166316;
    a2 = 0.2776316;
    a3 = 0.08357895;
    a4 = 0.00694737;
    psi = (2* (0:N1) * pi)/N1;
    flattopWin = a0-(a1*cos(psi))+(a2*cos(2*psi))-(a3*cos(3*psi))+(a4*cos(4*psi));
    flattopWin=flattopWin/sum(flattopWin);
    smoothEcg = conv(sigSquared,flattopWin,'same');
    
    % moving window integration
    winWidth2 = 150/1000;
    N2 = round(winWidth2*Fs);
    intEcg = zeros(size(smoothEcg));
    for n=1:length(smoothEcg)
        windowIndex = max(1,n-(N2-1)):n;
        intEcg(n) = mean(smoothEcg(windowIndex));
    end
    
%% Decision Phase
    [peaks1,peaks1locs,peaks1w] = findpeaks(filtsig,tm,'MinPeakDistance',231/1000);
    % change to indices
    for n=1:length(peaks1)
        peaks1(n)=find(tm==peaks1locs(n));
    end
 
    
    % initialising treshold
    firstInt = find(tm==2);
    maxF = max(filtsig(1:firstInt));
    meanF = mean(filtsig(1:firstInt));
    tresh1 = maxF/3;
    tresh2 = meanF/2;
    % initialising running signal and noise peak
    spk = tresh1;
    npk = tresh2;
    
    rpeaks = [];
    tpeaks=[];
    twidth=[];
    qrs=[];
    rpeaks2=[];
    for i=1:length(peaks1)
        if i>1
            tresh1 = npk+0.25*(spk-npk);
            tresh2 = 0.4*tresh1;
        end
        if filtsig(peaks1(i))>tresh1
            rpeaks=[rpeaks peaks1(i)];
            % RULE 1
            spk = 0.125*filtsig(rpeaks(end)) + 0.875*spk;
            npk = 0.125*filtsig(rpeaks(end)) + 0.875*npk;
            continue
        end
        
        if length(rpeaks)>8
            meanRR = mean(diff(tm(rpeaks(end-8:end))));
            RR = tm(peaks1(i))-tm(rpeaks(end));
            
            if (RR<0.36)||(RR<(0.5*meanRR))
                width = 70/1000;
                winStart = max(1,rpeaks(end)-round(width*Fs));
                winEnd = rpeaks(end)-1;
                meanSlopeR = mean(intEcg(winStart:winEnd));
                
                winStart = max(1,peaks1(i)-round(width*Fs));
                winEnd = peaks1(i)-1;
                meanSlope = mean(intEcg(winStart:winEnd));
                
                if meanSlope<(0.6*meanSlopeR)
                    tpeaks=[tpeaks peaks1(i)];
                    twidth = [twidth peaks1w(i)];
                    continue
                else
                    qrs=[qrs peaks1(i)];
                    % RULE 1
                    spk = 0.125*filtsig(rpeaks(end)) + 0.875*spk;
                    npk = 0.125*filtsig(rpeaks(end)) + 0.875*npk;
                    continue
                end
                
            else
                if (RR>1)||(RR>(1.66*meanRR))
                    if RR>1.4
                        if filtsig(peaks1(i))>(0.2*tresh2)
                            rpeaks2=[rpeaks2 peaks1(i)];
                            % RULE 2
                            spk = 0.75*filtsig(peaks1(i))+0.25*spk;
                            npk = 0.75*filtsig(peaks1(i))+0.25*npk;
                            continue
                        end
                    end 
            end
            end
        end
    end
    rpeaks = sort([rpeaks rpeaks2]);
    
%% Peaks detection based on R-peak Segmentation
    qpeaks=[];
    speaks=[];
    qrsLength=[];
    QTduration=NaN(1,length(rpeaks)-1);
    durationT=[];
    invertedfilt = -filtsig;
    
    for i=1:length(rpeaks)-1
        qIndex = max(1,rpeaks(i)-30);
        rIndex = rpeaks(i);
        sIndex = min(rpeaks(i)+30,length(filtsig));
        pIndex = max(1,rpeaks(i)-60);

        if pIndex>0 && sIndex<length(filtsig)
            % Detecting Q-peak
            [qp,ql,qw] = findpeaks(invertedfilt(qIndex:rIndex),tm(qIndex:rIndex),'MinPeakHeight',filtsig(rpeaks(i))*0.1,'MinPeakDistance',0.01);
            if ql
                indexQ = find(qp==max(qp));
                actualQindex = find(tm==ql(indexQ));
                qpeaks=[qpeaks actualQindex];
                
                % Check if T exists for the R-peak
                tp = tpeaks(tpeaks>rIndex&tpeaks<rpeaks(i+1));
                if tp
                    % Calculate QT duration
                   QT = tm(tp(1))-tm(actualQindex); 
                   QTduration(1,i)=QT;
                end
            end
        end
    end 
  PVCScore = length(PVC)/length(rpeaks);
  meanQTinterval = mean(QTduration);