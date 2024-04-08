function CI=calculateCI(sinus_time, ectopic_times)
    CI=arrayfun(@(et) et -sinus_time(find(sinus_time< et, 1, 'last')), ectopic_times);
end 