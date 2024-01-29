clear all;close all;clc;
load('30m.mat')
input = val(1,:);
fsamp = 250;
L = length(input);
time_ax = (0:1/fsamp:(L-1)/fsamp);

%% Filtering input
ecgfft = fft(input);
freq_ax = fsamp/L*(0:L-1);
freqIn = find(freq_ax==6);
freqOut = find(freq_ax==15);
ecgfft(1:freqIn)=0;
ecgfft(freqOut:end)=0;
ecg = real(ifft(ecgfft));

%% Calculating SNR
numerator = sum(input);
diff = [];
for k = 1:length(input)
    d = abs(input(k)-ecg(k));
    diff = [diff;d];
end
denominator = sum(diff.^2);

SNR = 10*log10(numerator/denominator);

%% Displaying filtered ECG
figure(1)
hold on
plot(time_ax,input)
plot(time_ax,ecg)

%% Processing
maxP = max(ecg);
passmark1 = 0.1*maxP;
peaks1 = [];
for i=1:length(ecg)
    if ecg(i)>passmark1
        peaks1 = [peaks1;ecg(i)];
    end
end

P_mean = mean(peaks1);
passmark2 = 0.45*P_mean;
peaks2 = [];
peaks2index = [];
for i=1:length(ecg)
    if ecg(i)>passmark2
        peaks2 = [peaks2;ecg(i)];
        peaks2index = [peaks2index;i];
    end
end

time_peaks2 = [];
for j = 1:length(peaks2index);
    time_peaks2 = [time_peaks2;time_ax(peaks2index(j))];
end


%% Calculating distance between consecutive R-peaks
[Rpeaks,indexR]= findpeaks(peaks2,time_peaks2,'MinPeakProminence',35,'MinPeakHeight',maxP/2);
Rintervals = [];

for n = 1:(length(Rpeaks)-1)
    d = indexR(n+1)-indexR(n);
    Rintervals = [Rintervals;d];
end
    
meanRint = mean(Rintervals);
bpm = 60/meanRint;

wind=[indexR;time_ax(end)];
Speaks =[];
indexS=[];
for k=1:1:(length(wind)-1)
    start = find(time_ax==wind(k));
    stop = find(time_ax==wind(k+1));
    ecgsamp = ecg(start:stop);
    timesamp = time_ax(start:stop);
    %plot(timesamp,ecgsamp)
    afterR = ecgsamp(1:round((length(ecgsamp)/2)));
    S_amp = min(afterR);
    Speaks = [Speaks;S_amp];
    S_ind = timesamp(find(ecgsamp==S_amp));
    indexS = [indexS;S_ind];
end

%% Detecting Q peaks
wind2 = [0;wind];
Qpeaks =[];
indexQ=[];
for k=1:1:(length(wind2)-1)
    start = find(time_ax==wind2(k));
    stop = find(time_ax==wind2(k+1));
    ecgsamp = ecg(start:stop);
    timesamp = time_ax(start:stop);
    beforeR = ecgsamp(round((length(ecgsamp)/2)):end);
    Q_amp = min(beforeR);
    Qpeaks = [Qpeaks;Q_amp];
    Q_ind = timesamp(find(ecgsamp==Q_amp));
    indexQ = [indexQ;Q_ind];
end

%% Detecting T-peaks
Tpeaks =[];
indexT=[];
index = sort([indexQ;indexS]);
for k=1:1:(length(index)-1)
    if (index(k+1)-index(k))<0.2
        continue
    end
    start = find(time_ax==index(k));
    stop = find(time_ax==index(k+1));
    ecgsamp = ecg(start:stop);
    timesamp = time_ax(start:stop);
    afterR = ecgsamp(1:round((length(ecgsamp)/2)));
    T_amp = max(afterR);
    Tpeaks = [Tpeaks;T_amp];
    T_ind = timesamp(find(ecgsamp==T_amp));
    indexT = [indexT;T_ind];
end

%% Plotting ecg and annotate peaks
figure
hold on
plot(time_ax,ecg,'k')

plot(indexQ,Qpeaks,'x','Color',[0.9290 0.6940 0.1250])
text(indexQ,Qpeaks-3,'Q')

plot(indexR,Rpeaks,'x','Color',[0.6350 0.0780 0.1840])
text(indexR,Rpeaks+4,'R')

plot(indexS,Speaks,'x','Color',[0 0.4470 0.7410])
text(indexS,Speaks-3,'S')

plot(indexT,Tpeaks,'x','Color',[0.4940 0.1840 0.5560])
text(indexT,Tpeaks+4,'T')

