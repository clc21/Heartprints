function heartPrints = heartprints_opt(filename, rec_length)
% Initialize and load library
    [~, config]=wfdbloadlib;
    
    % Get sampling frequency and calculate number of samples
    record_info=wfdbdesc(filename);
    fs=double(record_info(1).SamplingFrequency);
    duration_seconds=rec_length*3600;
    N=duration_seconds*fs;
    
    % Read annotations and filter out invalid ones
    [annIndices, annTypes]=rdann(filename, 'ari', [], N, []);
    validIndices=annIndices(annTypes ~= '?'); 
    N0=validIndices(1)-1;
    
    % Read sample data starting from first valid annotation
    [~, ~, tm] = rdsamp(filename, [], N, N0, 1, []);

    beatTypes=['N', 'V', 'r', 'S']; 
    beatList=cell(1, length(beatTypes));
    beatTimes=cell(1, length(beatTypes));

    for i = 1:length(beatTypes)
        beatList{i} = annIndices(annTypes == beatTypes(i));
        beatTimes{i}=tm(beatList{i}-N0); 
    end
    ectopic_times=sort(vertcat(beatTimes{2:end}));
    
    sinus_intervals = diff(beatTimes{1});
    ventricular_intervals = diff(ectopic_times);
    
    % Apply thresholds
    sinus_intervals(sinus_intervals > 2.5) = [];
    ventricular_intervals(ventricular_intervals > 4) = [];
    sinus_time=beatTimes{1};
    % Extract Coupling Interval (CI) and NIB
    CI=calculateCI(sinus_time,ectopic_times);
    NIB=calculateNIB(sinus_time,ectopic_times);

    sinus_intervals = sinus_intervals(~isoutlier(sinus_intervals));
    [nn_edges_max,nn_edges,vv_edges,nib_edges,ci_edges,nn_vv,nn_nib,nn_ci]=calcHeartprint(sinus_intervals,ventricular_intervals,NIB,CI);

    
    % Generate and save heartprint plots
    heartPrints=generateHeartPrintPlots(sinus_intervals, ventricular_intervals, CI, NIB,nn_edges_max,nn_edges,vv_edges,nib_edges,ci_edges,nn_vv,nn_nib,nn_ci,filename,rec_length)
end

function CI=calculateCI(sinus_time, ectopic_times)
    CI=arrayfun(@(et) et -sinus_time(find(sinus_time< et, 1, 'last')), ectopic_times);
end 

function NIB = calculateNIB(sinus_time,ectopic_times)
    edges = [ectopic_times; inf];
    NIB_counts = histcounts(sinus_time, edges);
    NIB = min(NIB_counts, 10);
end

function [nn_edges_max,nn_edges,vv_edges,nib_edges,ci_edges,nn_vv,nn_nib,nn_ci]=calcHeartprint(sinus_intervals,ventricular_intervals,NIB,CI)
    sinus_intervals = sinus_intervals(~isoutlier(sinus_intervals));
    
    calculateBinWidth=@(data, numBins)(max(data)-min(data))/numBins;
    ci_width=calculateBinWidth(CI, length(CI));
    nib_width=calculateBinWidth(NIB,50);

    nn_edges_max=max(sinus_intervals)+0.5;
    nn_edges=0:0.03:nn_edges_max;
    vv_edges=0:0.01:4; 
    nib_edges=0:nib_width:max(NIB)+nib_width;
    ci_edges=0:ci_width:max(CI)+ci_width;
    
    minLengthVV=min([length(sinus_intervals), length(ventricular_intervals)]);
    minLengthNIB=min([length(sinus_intervals), length(NIB)]);
    minLengthCI=min([length(sinus_intervals),length(CI)]);
    ventricular_intervals=ventricular_intervals(1:minLengthVV);
    NIB = NIB(1:minLengthNIB);
    CI=CI(1:minLengthCI);

    
    nn_vv = histcounts2(ventricular_intervals(:), sinus_intervals(1:minLengthVV), vv_edges, nn_edges);
    nn_nib = histcounts2(NIB(:), sinus_intervals(1:minLengthNIB), nib_edges, nn_edges);
    nn_ci = histcounts2(CI(:), sinus_intervals(1:minLengthCI), ci_edges, nn_edges);

end 


function heartPrints = generateHeartPrintPlots(sinus_intervals, ventricular_intervals, CI, NIB,nn_edges_max,nn_edges,vv_edges,nib_edges,ci_edges,nn_vv,nn_nib,nn_ci,filename,rec_length)
    heartPrints=figure(1);
    set(heartPrints,'position',[100 200 1300 350],'color','w','Name',['Heartprints for ' filename ':' num2str(rec_length) 'hours'])
    sgtitle(heartPrints, ['Heartprints for ' filename ': ' num2str(rec_length) ' hours'])
    
    subplot(245)
    num_bins_nn=ceil((max(sinus_intervals)-min(sinus_intervals))/0.03);    % Calculate the number of bins based on the desired bin width
    histogram(sinus_intervals, num_bins_nn, 'Orientation', 'horizontal');
    ylim([0 nn_edges_max])
    set(gca, 'XDir', 'reverse')
    ylabel('NN interval (s)'); 
    
    subplot(242)
    num_bins_vv=ceil((max(ventricular_intervals)-min(ventricular_intervals))/0.01); 
    histogram(ventricular_intervals,num_bins_vv);
    xlim([0 4])
    
    subplot(243)
    histogram(NIB)
    
    subplot(244)
    histogram(CI,length(CI))
    xlim([0 2.5])
    
    subplot(246)
    imagesc(vv_edges(1:end),nn_edges(1:end),nn_vv')
    xlabel('VV Interval Time (s)');
    xlim([0 4])
    ylim([0 nn_edges_max])
    axis xy;colormap('jet');
    
    subplot(247)
    imagesc(nib_edges(1:end),nn_edges(1:end), nn_nib');
    xlabel('NIB Interval Time (s)');
    ylim([0 nn_edges_max])
    axis xy;
    colormap('jet');
    
    subplot(248)
    imagesc(ci_edges(1:end),nn_edges(1:end),nn_ci');
    xlabel('CI Interval Time (s)');
    xlim([0 2.5])
    ylim([0 nn_edges_max])
    axis xy;
    colormap('jet');
    
    % Save figure to file
    saveHeartPrintToFile(heartPrints, filename, rec_length);
end

function saveHeartPrintToFile(fig, filename, rec_length)
    MyPath='C:\Users\User\OneDrive\Desktop\Year3_local';
    FiguresDir='Figures optimised'; 
    FigPath=fullfile(MyPath, FiguresDir); 
    if ~isfolder(FigPath)
        mkdir(FigPath);
    end
    [~, baseFileName, ~]=fileparts(filename); 
    FigFile=fullfile(FigPath, [baseFileName '_' num2str(rec_length) '.png']);
    exportgraphics(fig, FigFile,'Resolution', 600);
end
