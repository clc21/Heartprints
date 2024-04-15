clc;clear all;close all;

%% Generate sinus beats
theta = 0.4;
increment = 1/128;

% generate sinus intervals in the range of [0.5, 1.25]s
sinus_interval = 0.5:increment:1.25;
sinus_intervals = repelem(sinus_interval,1000)';  % to make up 24hours record length

% get sinus_times
sinus_time = 0;
sinus_times = [];

for i = 1:length(sinus_intervals)
    sinus_time = sinus_time+sinus_intervals(i);
    sinus_times = [sinus_times;sinus_time];
end

%% Model 3: Independent pacemaker
tv=1.75;                                        % 1.75s ventricular cycle
record_length = 24*3600;         
ectopic_times = transpose(0:tv:record_length);
ventricular_intervals = tv*ones(1, length(ectopic_times));

%% Refractory period
% theta = 0.4;

% if the V beat is expressed during the refractory period of the sinus
% cycle, then the V beat will not be expressed.
% similarly, if the N beat is expressed during the refractory period of the
% V beat, then the N beat will not be expressed.

ectopic_times_inRefrac = false(size(ectopic_times));

for i = 1:numel(ectopic_times)
    ectopic_times_inRefrac(i) = any(ectopic_times(i)>=sinus_times & ectopic_times(i)<=sinus_times+0.4);
end

% Remove elements if they fall within the ranges
ectopic_times(ectopic_times_inRefrac) = [];

ventricular_intervals = diff(ectopic_times);
ventricular_intervals = ventricular_intervals(ventricular_intervals>=theta);

%% CI

CI=[];
for i = 1:length(ectopic_times)
    ectopic_time=ectopic_times(i);          
    prev_N=find(sinus_times<ectopic_time);
    if ~isempty(prev_N)                       
        prevN_beat=sinus_times(prev_N(end));
        ci=ectopic_time-prevN_beat;      
        CI=[CI; ci];                
    end
end

% CI=arrayfun(@(et) et -sinus_times(find(sinus_times< et, 1, 'last')), ectopic_times, 'UniformOutput', false);


%% NIB
% NIB=[]; 
% 
% for i=1:(length(ectopic_times)-1)
%     current_V=ectopic_times(i);
%     next_V=ectopic_times(i+1);
% 
%     intervening_V=find(sinus_times>current_V & sinus_times<next_V);
%     nib=length(intervening_V);
%     nib=nib(nib<=15);
% 
%     NIB=[NIB; nib]; 
% end

edges = [ectopic_times; inf];
NIB_counts = histcounts(sinus_times, edges);
NIB = min(NIB_counts, 15);


%% Plotting

ci_width=(max(CI)-min(CI))/length(CI);

nn_edges_max=1.25;
nn_edges=0:increment:1.5;
vv_edges=0:0.04:max(ventricular_intervals); 
nib_edges=0:1:max(NIB)+1;
ci_edges=0:ci_width:max(CI)+ci_width;

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
ylim([0.5 0.70])
axis xy;colormap('jet');
set(gca, 'YTickLabel', {' '});

subplot(247)
imagesc(nib_edges(1:end),nn_edges(1:end), nn_nib');
xlabel('NIB Interval Time (s)');
ylim([0.5 0.70])
axis xy;
colormap('jet');
set(gca, 'YTickLabel', {' '});


subplot(248)
imagesc(ci_edges(1:end),nn_edges(1:end),nn_ci');
xlabel('CI Interval Time (s)');
xlim([0 1])
ylim([0.5 0.70])
axis xy;
colormap('jet');
set(gca, 'YTickLabel', {' '});