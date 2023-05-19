rng(1); % set random seed

% pathы to the folder with data
subjs = string(cellstr(ls('D:\WorkFolder\MasterThesis\data\data_exp\*_*')));
subjs_pars = string(cellstr(ls('D:\WorkFolder\MasterThesis\data\data_params\*_*')));

for i = 1:length(subjs_pars)
    df = 'D:\WorkFolder\MasterThesis\data\data_params\' + subjs_pars(i);
    sp1 = split(subjs_pars(i), '.');
    sp2 = split(sp1(1), '_');
    day = sp2(3);
    idd = sp2(1) + '_' + sp2(2);

    best_params = load(df).best_params;

    n_day = string(extract(day,2));
    folder = 'D:\WorkFolder\MasterThesis\data\data_exp\' + idd;
    [stim_seq, rewards, episodes_inds] = extract_data(folder, n_day);

    [act_val, act_seq] = simulate_data(stim_seq, rewards, best_params);

    % save results as a structure
    to_save = struct('best_params', best_params, ...
        'episodes_inds', episodes_inds, ...
        'act_val', act_val, 'act_seq', act_seq, 'rewards', rewards);
    save(idd + '_' + day + "_sim.mat", "-struct", "to_save");
end


% extract participant's data
function [stim_seq, rewards, episodes_inds] = extract_data(folder, n_day)
    % list of files with behavoural data, 6 runs per subject for single day
    n_day = string(n_day);
    d = char(folder + '/day_' + n_day + '/');
    day_name = char(folder + '/day_' + n_day + '/*.mat');
    day = string(cellstr(ls(day_name)));

    % iterate over behavoral data files and extract necessary data
    for j = 1:length(day)
        path = d + day(j);
        data = load(path); % load single file
        mat = data.data.matrix; % load matrix from data
    end

    % save stimuli sueqence and episodes sequence
    stim_seq = mat(:, 3);
    episodes = mat(:, 1);

    % save expected answers and build rewards matrix
    exp_ans = mat(:, 4);
    rewards = zeros(length(exp_ans), 4);
    for i = 1:length(exp_ans)
        rewards(i, exp_ans(i)) = 1;
    end

    % get indices of episodes
    episodes_inds = zeros(24, 1);
    for i = 1:24
        ind = find(episodes == i, 1, 'first');
        episodes_inds(i) = ind;
    end

end


% simulate data given stimuli, rewards and parameters
function [act_val, act_seq] = simulate_data(stim_seq, rewards, cur_params)

    % some default parameters for all models
    nS = 3;
    nC = 0;
    nA = 4;
    ctxt_seq = [];
    act_seq = [];

    % start parameters
    params = [cur_params(1), cur_params(2), 0, 1, 0, cur_params(3), ...
        cur_params(4), cur_params(5), cur_params(6), cur_params(7), 1, 0, 0];

    % model with start parameters
    [act_val, act_seq, ~, ~] = CLEF_model_test(nS, nC, stim_seq, ...
        ctxt_seq, act_seq, rewards, params);
end




% какие параметры оптимизируем - (1) mon_size, (2) rl_alpha, (6) softmax_beta,
% (7) softmax_eps, (8) volatility, (9) bias_conf, (10) bias_ini
%
% какие параметры дефолт/не нужны - (3) rl_decay 0, (4) bay_prior 1,
% (5) bay_decay 0, (11) ctxt_prior 1, (12) ctxt_decay 0, (13) bias_ctxt 0
