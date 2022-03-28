if ~exist('model','var')
    error("Please define model ('model' variable) before running the script!'");
end

full_provider = MultiDistanceProvider();
min_max_provider = MultiDistanceProvider();
min_max_provider.Settings.UseLinf = false;
min_max_provider.Settings.UseW1inf = false;
min_max_provider.Settings.UseW1infPrime = false;

% NO NORMALIZATION
model.DistanceProvider = full_provider;

% N = 50, min-max normalisation
sref = model.simulate(50);
run one_pattern_template

% N = 25, min-max normalisation
sref = model.simulate(25);
run one_pattern_template

% N = 10, min-max normalisation
sref = model.simulate(10);
run one_pattern_template

% MIN-MAX NORMALIZATION
model.Observer = MinMaxObserver();
model.DistanceProvider = min_max_provider;

% N = 50, min-max normalisation
sref = model.simulate(50);
run one_pattern_template

% N = 25, min-max normalisation
sref = model.simulate(25);
run one_pattern_template

% N = 10, min-max normalisation
sref = model.simulate(10);
run one_pattern_template