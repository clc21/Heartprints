function [sinus_intervals,ventricular_intervals,NIB,CI]= initHP(filename,start_time,end_time,fs,N0)

    % Determine location to start sampling for given segment
    N1=(start_time*3600*fs)+N0;

    % Determine location to end sampling for given segment
    if end_time == 24
        N=[];
    else
        N=N1+(28800*fs);
    end

    % Read annotations and filter out invalid ones
    [annIndices, annTypes]=rdann(filename, 'ari',[],N,N1,[]);

    % Read sample data starting from first valid annotation
    [~, ~, tm] = rdsamp(filename, [], N, N1, 1, []);
    beatTypes=['N', 'V', 'r', 'S']; 
    beatList=cell(1, length(beatTypes));
    beatTimes=cell(1, length(beatTypes));

    % Categorise beat types
    for i = 1:length(beatTypes)
        beatList{i} = annIndices(annTypes == beatTypes(i));
        beatTimes{i}=tm(beatList{i}-N1); 
    end
    ectopic_times=sort(vertcat(beatTimes{2:end}));
    
    % Calculate intervals
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
