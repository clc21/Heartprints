function CI=calculateCI(sinus_time, ectopic_times)
    CI=arrayfun(@(et) et-sinus_time(find(sinus_time< et, 1, 'last')), ectopic_times,'UniformOutput',false);
    CI=cell2mat(CI(~cellfun('isempty', CI)));
end 
