function [sinus_intervals,ventricular_intervals,NIB,CI]= initHP(filename,start_time,end_time,fs)

    duration_seconds=(end_time - start_time) * 3600;
    N1=start_time*3600*fs;

    if end_time == 24
        N=[];
    
    else
    N=N1+(duration_seconds*fs);
    end

    % Read annotations and filter out invalid ones
    [annIndices, annTypes]=rdann(filename, 'ari',[],N,N1,[]);
    validIndices=annIndices(annTypes ~= '?'); 
    N0=validIndices(1)-1;

    if N1>N0
        N0=N1;
    else
    end

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
    
end