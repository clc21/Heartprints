function [nn_edges_max,nn_edges,vv_edges,nib_edges,ci_edges,nn_vv,nn_nib,nn_ci]=calcHeartprint(combinedData)
    sinus_intervals=combinedData.sinus_intervals;
    ventricular_intervals=combinedData.ventricular_intervals;
    CI=combinedData.CI;
    NIB=combinedData.NIB;
    
    % Determine appropriate bin width and edges
    calculateBinWidth=@(data, numBins)(max(data)-min(data))/numBins;
    ci_width=calculateBinWidth(CI, length(CI));

    nn_edges_max=max(sinus_intervals)+0.5;
    nn_edges=0:0.03:nn_edges_max;
    vv_edges=0:0.01:4; 
    nib_edges=0:1:8;
    ci_edges=0:ci_width:max(CI)+ci_width;

    % Concatenate data to allow for same length
    minLengthVV=min([length(sinus_intervals), length(ventricular_intervals)]);
    minLengthNIB=min([length(sinus_intervals), length(NIB)]);
    minLengthCI=min([length(sinus_intervals),length(CI)]);
    ventricular_intervals=ventricular_intervals(1:minLengthVV);
    NIB=NIB(1:minLengthNIB);
    CI=CI(1:minLengthCI);

    % Generate 2D histogram 
    nn_vv=histcounts2(ventricular_intervals(:), sinus_intervals(1:minLengthVV), vv_edges, nn_edges);
    nn_nib=histcounts2(NIB(:), sinus_intervals(1:minLengthNIB), nib_edges, nn_edges);
    nn_ci=histcounts2(CI(:), sinus_intervals(1:minLengthCI), ci_edges, nn_edges);

end
