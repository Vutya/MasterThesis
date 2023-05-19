rng(1); % set random seed

% path to the folder with all data
subjs = string(cellstr(ls('D:\WorkFolder\MasterThesis\data\data_exp\*_*')));

% iteration over files
for i = 1:length(subjs)
    % path to the folder of a single subject
    folder = 'D:\WorkFolder\MasterThesis\data\data_exp\' + subjs(i);
    disp(folder);
    
    % iteration over days of the subject
    for day = 1:2
        % get all participants data
        [stim_seq, act_seq, rewards, correct, incorrect, missed, total] = extract_data(folder, day);

        % fit parameters with genetic algorithms
        [ga_params, ga_LL, act_val, act_seq, ltm, d] = fit_ga(stim_seq, act_seq, rewards);
        disp(ga_LL);

        ps_size = int64(ga_params(1)); % save monitoring buffer size

        % check points from 1 to mon_size (around the minimum)
        comb_params = zeros(1,7);
        for j = 1:ps_size
            comb_params(j,:) = ga_params;
            comb_params(j,1) = j;
        end
      
        % also check points with slightly changed softmax beta (+- 0.5 from the minimum)
        for k = 1:2
            for j = 1:ps_size
                comb_params(k*ps_size + j, :) = comb_params(j,:);
                comb_params(k*ps_size + j, 3) = comb_params(j,3) + (2/3) * comb_params(j,3) * (-1)^k;
            end
        end

        % save results of ga as the current best
        best_params = ga_params;
        best_LL = ga_LL;

        % check additional pounts around minimum with pattern search
        for t = 1:size(comb_params, 1)
            start_params = comb_params(j, :);

            [fit_params, LL, act_val_ps, act_seq_ps, ltm_ps, d_ps] = fit_ps(stim_seq, act_seq, rewards, start_params);
            if LL > best_LL
                best_LL = LL;
                best_params = fit_params;
                act_val = act_val_ps;
                act_seq = act_seq_ps;
                ltm = ltm_ps;
                d = d_ps;
            end
        end

        % save results as a structure
        to_save = struct('best_params', best_params, 'best_LL', best_LL, ...
            'ga_params', ga_params, 'ga_LL', ga_LL, ...
            'correct', correct, 'incorrect', incorrect, 'missed', missed, 'total', total, ...
            'act_val', act_val, 'act_seq', act_seq, 'ltm', ltm, 'd', d);
        save(subjs(i) + "_d" + day + ".mat", "-struct", "to_save");
    end
end


% extract participant's data
function [stim_seq, act_seq, rewards, correct, incorrect, missed, total] = extract_data(folder, n_day)
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
        % save sequences of answers and correctness
        if j == 1
            act_seq = zeros([length(mat), 1]);
            ans_seq = zeros([length(mat), 1]);
        end
        act_seq = act_seq + mat(:, 10);
        ans_seq = ans_seq + mat(:, 12);
    end

    % save stimuli sueqence
    stim_seq = mat(:, 3);

    % save expected answers and build rewards matrix
    exp_ans = mat(:, 4);
    rewards = zeros(length(exp_ans), 4);
    for i = 1:length(exp_ans)
        rewards(i, exp_ans(i)) = 1;
    end

    % save number of total, correct, incorrect, and missed answers
    total = length(ans_seq);
    correct = sum(ans_seq(:) == 1);
    incorrect = sum(ans_seq(:) == 0);
    missed = sum(ans_seq(:) == -1);

end


% Compute Negative LogLikelihood
function NegLL = lik(stim_seq, act_seq, rewards, cur_params)

    % some default parameters for all models
    nS = 3;
    nC = 0;
    nA = 4;
    ctxt_seq = [];

    % start parameters
    params = [cur_params(1), cur_params(2), 0, 1, 0, cur_params(3), ...
        cur_params(4), cur_params(5), cur_params(6), cur_params(7), 1, 0, 0];

    % model with start parameters
    [act_val, ~, ~, ~] = CLEF_model_test(nS, nC, stim_seq, ...
        ctxt_seq, act_seq, rewards, params);

    % compute negative sum of loglikelihoods
    ll_probs = ones([length(act_val) 1]);

    for t = 1:length(act_seq)
        if act_seq(t) ~= 0  
            act_probs = exp(cur_params(3)*(act_val(t,:)- max(act_val(t,:))));
            act_probs = cur_params(4)/nA + (1-cur_params(4))*act_probs/sum(act_probs);
            ll_probs(t) = act_probs(act_seq(t));
        end
    end
    NegLL = -sum(log(ll_probs));
end


% fit model with behavioral data and genetic algorithms
function [fitParams, LL, act_val, act_seq, ltm, d] = fit_ga(stim_seq, act_seq, rewards)
    obFunc = @(x) lik(stim_seq, act_seq, rewards, x); % lambda function

    % lower and upper bounds
    LB = [1 0 0 0 0 0 0];
    UB = [10 1 80 0.5 1 1 1];

    % set options and run ga
    options = optimoptions('ga', 'UseParallel', true, 'PopulationSize', 24000);
    [fitParams, NegLL] = ga(obFunc, 7, [], [], [], [], LB, UB, [], options);

    LL = -NegLL; % LL with minus

    % some default parameters for all models
    nS = 3;
    nC = 0;
    ctxt_seq = [];

    % start parameters
    params = [fitParams(1), fitParams(2), 0, 1, 0, fitParams(3), ...
        fitParams(4), fitParams(5), fitParams(6), fitParams(7), 1, 0, 0];

    % model with start parameters
    [act_val, act_seq, ltm, d] = CLEF_model_test(nS, nC, stim_seq, ...
        ctxt_seq, act_seq, rewards, params);
end


% fit model with behavioral data and pattern search
function [fitParams, LL, act_val, act_seq, ltm, d] = fit_ps(stim_seq, act_seq, rewards, cur_params)
    obFunc = @(x) lik(stim_seq, act_seq, rewards, x);  % lambda function

    % lower and upper bounds
    LB = [1 0 0 0 0 0 0];
    UB = [inf 1 inf 0.5 1 1 1];

    % set options and run pattern search
    options = optimoptions('patternsearch', 'UseParallel', true, ...
        'AccelerateMesh', true, 'InitialMeshSize', 0.5, ...
        'MaxFunctionEvaluations', 16000, 'MaxIterations', 1000, ...
        'MeshTolerance', 1e-7, 'StepTolerance', 1e-7, ...
        'UseCompletePoll',true, 'UseCompleteSearch', true);
    [fitParams, NegLL] = patternsearch(obFunc, cur_params, [], [], [], [], LB, UB, [], options);

    LL = -NegLL; % LL with minus

    % some default parameters for all models
    nS = 3;
    nC = 0;
    ctxt_seq = [];

    % start parameters
    params = [fitParams(1), fitParams(2), 0, 1, 0, fitParams(3), ...
        fitParams(4), fitParams(5), fitParams(6), fitParams(7), 1, 0, 0];

    % model with start parameters
    [act_val, act_seq, ltm, d] = CLEF_model_test(nS, nC, stim_seq, ...
        ctxt_seq, act_seq, rewards, params);
end


% какие параметры оптимизируем - (1) mon_size, (2) rl_alpha, (6) softmax_beta,
% (7) softmax_eps, (8) volatility, (9) bias_conf, (10) bias_ini
%
% какие параметры дефолт/не нужны - (3) rl_decay 0, (4) bay_prior 1,
% (5) bay_decay 0, (11) ctxt_prior 1, (12) ctxt_decay 0, (13) bias_ctxt 0
