%version 29/1 -- path to save fig is set to sofia's folder

function heartPrints=heartprintfunc(filename,N0,N)
    [~,config]=wfdbloadlib;
    [~,~,tm]=rdsamp(filename,[],N,N0,1,[]);
    
    %Extract sinus intervals & V-V interval
    [sinus_beats]=rdann(filename,'ari',[],N,N0,'N');
    [ectopic_beats]=rdann(filename,'ari',[],N,N0,'V');
    [r_beats]=rdann(filename,'ari',[],N,N0,'r');
    [S_beats]=rdann(filename,'ari',[],N,N0,'S');
    
    [ectopic_beats]=[ectopic_beats;r_beats];    
    [ectopic_beats]=[ectopic_beats;S_beats];
    ectopic_beats=sort(ectopic_beats);
    
    sinus_times=tm(sinus_beats-N0);                                     
    ectopic_times=tm(ectopic_beats-N0);                                 
    
    sinus_intervals=diff(sinus_times);                                  
    sinus_intervals=sinus_intervals(sinus_intervals<=2.5);
    
    ventricular_intervals=diff(ectopic_times);                           
    ventricular_intervals=ventricular_intervals(ventricular_intervals<=4);
    CI=[];

    % Extract Coupling Interval (CI)
    for i=1:length(ectopic_times)
        ectopic_time=ectopic_times(i);          
        prev_N=find(sinus_times<ectopic_time);
    
        if ~isempty(prev_N)                       
            prevN_beat=sinus_times(prev_N(end));
            
            ci=ectopic_time-prevN_beat;      
            CI=[CI; ci];                 
    
        end
    end

    %Extract NIB -  number of intervening sinus beats between each two successive ectopic beats
    NIB=[]; 
    
    for i=1:(length(ectopic_times)-1)
        current_V=ectopic_times(i);
        next_V=ectopic_times(i+1);
        
        intervening_V=find(sinus_times>current_V & sinus_times<next_V);
        nib=length(intervening_V);
        nib=nib(nib<=10);
    
        NIB=[NIB; nib]; 
    end
    
    %Heartprint initialising
    nn_outlier=isoutlier(sinus_intervals,'mean');
    sinus_intervals(nn_outlier)=0;

    hist_VV=histogram(ventricular_intervals,length(ventricular_intervals),'Visible', 'off');
    vv_width=hist_VV.BinWidth;
    nn_width=vv_width;
    
    hist_CI=histogram(CI,length(CI),'Visible', 'off');
    ci_width=hist_CI.BinWidth;
    nn_widthci=ci_width;
    
    hist_NIB=histogram(NIB,length(NIB),'Visible', 'off');
    nib_width=hist_NIB.BinWidth;
    nn_widthnib=nib_width; 
    
    minLengthVV=min(length(sinus_intervals), length(ventricular_intervals));
    sinus_intervals_VV=sinus_intervals(1:minLengthVV);
    ventricular_intervals_shortened=ventricular_intervals(1:minLengthVV);
    
    nn_edges=0:nn_width:2.5; 
    vv_edges=0:vv_width:max(ventricular_intervals);
    
    nn_vv=histcounts2(ventricular_intervals_shortened, sinus_intervals_VV, vv_edges, nn_edges);
    
    minLengthNIB=min(length(NIB),length(sinus_intervals));
    sinus_intervals_NIB=sinus_intervals(1:minLengthNIB);
    NIB_shortened=NIB(1:minLengthNIB);
    
    nn_edgesnib=0:nn_widthnib:2.5;
    nib_edges=0:nib_width:max(NIB);
    
    nn_nib=histcounts2(NIB_shortened,sinus_intervals_NIB,nib_edges,nn_edgesnib);
    
    minLengthCI=min(length(CI),length(sinus_intervals));
    sinus_intervals_CI=sinus_intervals(1:minLengthCI);
    CI_shortened=CI(1:minLengthCI);
    
    nn_edgesci=0:nn_widthci:2.5;
    ci_edges=0:nib_width:max(CI);
    
    nn_ci=histcounts2(CI_shortened,sinus_intervals_CI,ci_edges,nn_edgesci);

    if N==911027
        rec_length='1 hour';
    elseif N==1811063
        rec_length='2 hours';
    elseif N==5411143
        rec_length='6 hours';
    end 


    %Plot heartprints
    heartPrints=figure(1);
    set(heartPrints,'position',[100 200 1300 350],'color','w','Name',['Heartprints for ' filename ' ' rec_length])
    sgtitle(heartPrints, ['Heartprints for ' filename ' ' rec_length])
    
    subplot(245)
    histogram(sinus_intervals,length(sinus_intervals),'Orientation','horizontal');
    title('NN histogram');
    set(gca,'XDir','reverse')
    ylim([0 2.5]);
    xlabel('Count'); 
    ylabel('NN interval (s)'); 
    
    subplot(242)
    histogram(ventricular_intervals,length(ventricular_intervals))
    title('VV histogram')
    xlim([0 4])
    
    subplot(243)
    histogram(CI,length(CI))
    title('CI histogram')
    xlim([0 2.5])
    
    subplot(244)
    histogram(NIB, length(NIB))
    title('NIB histogram')
    
    subplot(246)
    imagesc(vv_edges,nn_edges,nn_vv')
    xlabel('VV Interval Time (s)');
    ylabel('NN Interval Time (s)');
    title('Heatmap of NN vs VV Intervals');
    axis xy;colormap('jet');
    xlim([0 4]);
    
    subplot(247)
    imagesc(ci_edges,nn_edgesci,nn_ci');
    xlabel('CI Interval Time (s)');
    ylabel('NN Interval Time (s)');
    title('Heatmap of NN vs CI Intervals');
    axis xy;xlim([0 2.5]);
    colormap('jet');
     
    subplot(248)
    imagesc(nib_edges,nn_edgesnib, nn_nib');
    xlabel('NIB Interval Time (s)');
    ylabel('NN Interval Time (s)');
    title('Heatmap of NN vs NIB Intervals');
    axis xy;
    colormap('jet');
    colorbar;

    % Export and save figure
    MyPath='C:\Users\User\OneDrive - Imperial College London\YEAR 3\DAPP3\heartprints'; 
    FiguresDir='Figures'; 
    FigPath=fullfile(MyPath, FiguresDir); 
    if ~isfolder(FigPath)
        mkdir(FigPath);
    end
    [~, baseFileName, ~]=fileparts(filename); 
    FigFile=fullfile(FigPath, [baseFileName '_' rec_length '.png']);
    exportgraphics(heartPrints, FigFile,'Resolution', 600);

end
