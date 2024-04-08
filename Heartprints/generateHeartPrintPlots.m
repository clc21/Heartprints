
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
