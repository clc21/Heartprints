function combinedData = mergeSegmentData(sinus_intervals,ventricular_intervals,NIB,CI)
    combinedData=struct('sinus_intervals', [], 'ventricular_intervals', [], 'CI', [], 'NIB', [], 'tm', []);
    
    combinedData.sinus_intervals=[combinedData.sinus_intervals;sinus_intervals];
    combinedData.ventricular_intervals=[combinedData.ventricular_intervals;ventricular_intervals];
    combinedData.NIB=[combinedData.NIB;NIB];
    combinedData.CI=[combinedData.CI;CI];


end
