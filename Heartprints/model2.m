clc;clear all;close all;

%% Generate sinus beats

increment = 1/128;

% generate sinus intervals in the range of [0.5, 1.25]s
sinus_interval = 0.5:increment:1.25;
num_sinus_interval = numel(sinus_interval);       % 97 resolutions
sinus_intervals = repelem(sinus_interval,1000)';  % to make up 24hours record length

sum_record_length = sum(sinus_interval);          % 84875 s , 23.5764 h -> almost 24 h

% get sinus_times
sinus_time = 0;
sinus_times = [];

for i = 1:length(sinus_intervals)
    sinus_time = sinus_time+sinus_intervals(i);
    sinus_times = [sinus_times;sinus_time];
end

%% Model 2: Fixed CI

% generate random ectopic beats
ci = 0.6;                                       % coupling interval fixed at 0.6s
record_length = 24*3600;         
p = 0.36;                                       % probability for an ectopic beat
n_V_beats = p*record_length;                    % number of ectopic beats occruring in 24h record length
ectopic_times = sort(rand(n_V_beats,1));
ectopic_times = ectopic_times.*record_length;         % time stamp of V beat occurence

% Refractory period
theta = 0.4;

ectopic_times_inRefrac = false(size(ectopic_times));

for i = 1:numel(ectopic_times)
    ectopic_times_inRefrac(i) = any(ectopic_times(i)>=sinus_times & ectopic_times(i)<=sinus_times+0.4);
end
ectopic_times(ectopic_times_inRefrac) = [];

% find ectopic beats and set CI
CI=[];

for i = 1:length(ectopic_times)
    ectopic_time=ectopic_times(i);          
    prev_N=find(sinus_times<ectopic_time);

    if ~isempty(prev_N)                       
        prevN_beat=sinus_times(prev_N(end));
        
        ectopic_times(i) = prevN_beat + ci;
        CI=[CI; ci];                 

    end
end

ventricular_intervals = diff(ectopic_times);
ventricular_intervals = ventricular_intervals(ventricular_intervals>=theta);

%% NIB
% NIB=[]; 
% 
% for i=1:(length(ectopic_times)-1)
%     current_V=ectopic_times(i);
%     next_V=ectopic_times(i+1);
% 
%     intervening_V=find(sinus_times>current_V & sinus_times<next_V);
%     nib=length(intervening_V);
%     nib=nib(nib<=10);
% 
%     NIB=[NIB; nib]; 
% end

edges = [ectopic_times; inf];
NIB_counts = histcounts(sinus_times, edges);
NIB = min(NIB_counts, 15);


%% Plotting

% calculateBinWidth=@(data, numBins)(max(data)-min(data))/numBins;
% ci_width=calculateBinWidth(CI, length(CI));

nn_edges_max=1.25;
nn_edges=0.5:increment:max(sinus_intervals);
vv_edges=0:0.01:max(ventricular_intervals); 
nib_edges=0:1:max(NIB)+1;
ci_edges=0:0.1:1.0;

minLengthVV=min([length(sinus_intervals), length(ventricular_intervals)]);
minLengthNIB=min([length(sinus_intervals), length(NIB)]);
minLengthCI=min([length(sinus_intervals), length(CI)]);
ventricular_intervals=ventricular_intervals(1:minLengthVV);
NIB = NIB(1:minLengthNIB);
CI=CI(1:minLengthCI);

nn_vv = histcounts2(ventricular_intervals(:), sinus_intervals(1:minLengthVV), vv_edges, nn_edges);
nn_nib = histcounts2(NIB(:), sinus_intervals(1:minLengthNIB), nib_edges, nn_edges);
nn_ci = histcounts2(CI(:), sinus_intervals(1:minLengthCI), ci_edges, nn_edges);


f1=figure(1);

subplot(245)
histogram(sinus_intervals, length(sinus_intervals), 'Orientation', 'horizontal');
ylim([0.5 nn_edges_max])
xlim([0 1100])
set(gca, 'XDir', 'reverse')
ylabel('NN interval (s)'); 

subplot(242)
num_bins_vv=ceil((max(ventricular_intervals)-min(ventricular_intervals))/0.01); 
histogram(ventricular_intervals,num_bins_vv);
xlim([0 10])

subplot(243)
histogram(NIB)

subplot(244)
histogram(CI,length(CI))
xlim([0 1])

subplot(246)
imagesc(vv_edges(1:end),nn_edges(1:end),nn_vv')
xlabel('VV Interval Time (s)');
xlim([0 10])
ylim([0.5 0.63])
axis xy;colormap('jet');
set(gca, 'YTickLabel', {' '});

subplot(247)
imagesc(nib_edges(1:end),nn_edges(1:end), nn_nib');
xlabel('NIB Interval Time (s)');
ylim([0.5 0.63])
axis xy;
colormap('jet');
set(gca, 'YTickLabel', {' '});


subplot(248)
imagesc(ci_edges(1:end),nn_edges(1:end),nn_ci');
xlabel('CI Interval Time (s)');
xlim([0 1])
ylim([0.5 0.63])
axis xy;
colormap('jet');
set(gca, 'YTickLabel', {' '});

