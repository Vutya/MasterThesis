rng(1);  % set random seed

% parameters to recover
params = zeros(10, 7);
params(1, :) = [1, 0.8, 1, 0.05, 0.25, 0.56, 0.19];
params(2, :) = [1, 0.05, 20, 0, 0.75, 0.01, 0.01];
params(3, :) = [2, 0.2, 0.5, 0.05, 0.33, 0.89, 0.89];
params(4, :) = [2, 0.7, 30, 0, 0.44, 0.66, 0];
params(5, :) = [3, 0.1, 10, 0.04, 0.32, 0.19, 0.60];
params(6, :) = [3, 0.3, 50, 0.25, 0.12, 0.99, 0.13];
params(7, :) = [3, 0.8, 0.5, 0.01, 0.02, 0.06, 0.24];
params(8, :) = [4, 0.6, 10, 0.4, 0.23, 0.05, 0.4];
params(9, :) = [5, 0.6, 10, 0.4, 0.23, 0.05, 0.4];
params(10, :) = [6, 0.6, 10, 0.4, 0.23, 0.05, 0.4];

% path to the example file
path = '/home/vvtimokhov/data/17_ULMA/day_2/Mnemosyne_S34_run1.mat';
data = load(path); 
mat = data.data.matrix;

% save sequence of stimuli, correct answers and build rewards matrix
stim_seq = mat(:, 3);
exp_ans = mat(:, 4);
rewards = zeros(length(exp_ans), 4);
for i = 1:length(exp_ans)
    rewards(i, exp_ans(i)) = 1;
end

% iteration over parameters
for i = 1:length(params)
    cur_params = params(i, :); % current parameters
    [act_val_sim, act_seq_sim, NegLL] = simulate(stim_seq, rewards, cur_params); % simulate data
    real_LL = -NegLL; % save real loglikelihood

    % fit data with genetic algorithms
    [ga_params, ga_LL, act_val, act_seq, ltm, d] = fit_ga(stim_seq, act_seq_sim, rewards);
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
            comb_params(k*ps_size + j, :) = ga_params;
            comb_params(k*ps_size + j, 3) = ga_params(3) + 0.5 * ga_params(3) * (-1)^k;
        end
    end

    % save results of ga as the current best
    best_params = ga_params;
    best_LL = ga_LL;

    % check additional pounts around minimum with pattern search
    for t = 1:size(comb_params, 1)
        start_params = comb_params(t, :);
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
    to_save = struct('real_params', cur_params, 'real_LL', real_LL, ...
        'ga_params', ga_params, 'ga_LL', ga_LL, ...
        'best_params', best_params, 'best_LL', best_LL, ...
        'act_val', act_val, 'act_seq', act_seq, 'ltm', ltm,  'd', d, ...
        'act_val_sim', act_val_sim, 'act_seq_sim', act_seq_sim);
    
    save("params_recover_" + i + ".mat", "-struct", "to_save");
end



% simulate data given stimuli, rewards and parameters
function [act_val, act_seq, NegLL] = simulate(stim_seq, rewards, cur_params)

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
    UB = [30 1 100 0.5 1 1 1];

    % set options and run ga
    options = optimoptions('ga', 'UseParallel', true, 'PopulationSize', 12000);
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
