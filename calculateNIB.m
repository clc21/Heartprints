function NIB = calculateNIB(sinus_time,ectopic_times)
    edges = [ectopic_times; inf];
    NIB_counts = histcounts(sinus_time, edges);
    NIB = min(NIB_counts, 10);
end 