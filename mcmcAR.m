function [ind_iter] = mcmcAR(y)

    %%% Parameters %%%
    n = size(y, 1); % update n
    ngroups = 2;
    ncovs = 1;
    ntime = 14;
    nloop = 1000; % number of iterations
    nwarmup = 500;
    ind_iter = ones(n, nloop+1); % index for group membership in each loop
    
    
    %%% Housekeeping %%%
    % to store the alphas (slopes) in each loop
    alpha_iter = zeros(n, ncovs+1, nloop+1);
    
    % mu_alpha will be mean of coef
    mu_alpha_iter = zeros(ngroups, ncovs+1, nloop+1);
    
    % tau_alpha will be var of coef
    tau_alpha_iter = zeros(ngroups, ncovs+1, nloop+1);
    
    % to store sigmasq in each loop
    sigmasq_iter = zeros(ngroups, nloop+1);
    
    prior_prob_iter = ones(ngroups, nloop+1); % priors
    post_prob_iter = ones(n,ngroups,  nloop+1); % posteriors
    
    %%% Initializing some starting values for MCMC %%%
    u = rand(n, 1); % random starting value for proportion assumed in each group
    kk = u < 0.5; % generate a starting group for each group member 
    ind_iter(kk, 1) = 2; % update the first index with starting group
    
    % Intercept and slope starting values for each individual 
    for k = 1:n    
        y_temp = y(k, 2:ntime)';
        y_lag = y(k, 1:ntime-1)';
        xmat = [ones(ntime-1, 1), y_lag]; % design matrix 
        xdashx = xmat' * xmat; % covariance between timepoints
        alpha_iter(k, :, 1) = xdashx \ (xmat' * y_temp); % intercept & slope 
    end
    
    % Mu, Tau and Sigmasq starting values
    for j = 1:ngroups
        kk = find(ind_iter(:, 1) == j); % was: find(ind_true == j);
        mu_alpha_iter(j, 1, 1) = mean(alpha_iter(kk, 1, 1)); % mean intercept of group
        mu_alpha_iter(j, 2, 1) = mean(alpha_iter(kk, 2, 1)); % mean slope of group
        tau_alpha_iter(j, 1, 1) = var(alpha_iter(kk, 1, 1));
        tau_alpha_iter(j, 2, 1) = var(alpha_iter(kk, 2, 1));
        temp = y(kk, :);
        sigmasq_iter(j, 1) = var(temp(:));
        prior_prob_iter(j, 1) = 1 / ngroups;
        post_prob_iter(:, j, 1) = 1 / ngroups;
    end
    
    %c=n; % Is c used?
    fit_ind = zeros(n, ntime-1, nloop+1);                    % What is fit_ind?
    ydev = zeros(n, ntime-1);                                % What is ydev?
    prior_sig_a = 1;
    prior_sig_b = 1;
    prior_mu_mean = mean(alpha_iter(:, :, 1))';
    prior_mu_prec = diag(0.01 ./ var(alpha_iter(:, :, 1))'); % precision
    prior_tau_a = 1;
    prior_tau_b = 1;
    prior_tau_uplim = 4 * range(alpha_iter(:, :, 1)).^2;
    ylag = y(:, 1:ntime-1);


    %% Starting MCMC scheme
    for p = 1:nloop

        % Drawing alphas from a Normal
        for i = 1:n % for each individual (n = 1000)

            j = ind_iter(i, p);
            y_ind = y(i, 2:ntime)';
            xmat = [ones(ntime-1, 1), y(i, 1:ntime-1)']; % design matrix
            xdashx = xmat' * xmat; % Intercept and slope for each individual
            % draw a random mu and sigma from the same distribution 
            alpha_prior_mean = mu_alpha_iter(j, :, p)';
            alpha_prior_prec = diag(1 ./ tau_alpha_iter(j, :, p));
            alpha_like_mean = xdashx \ (xmat' * y_ind);
            alpha_like_prec = xdashx / sigmasq_iter(j,p);
            alpha_post_prec = alpha_like_prec + alpha_prior_prec;
            alpha_post_var = inv(alpha_post_prec);
            alpha_post_mean = alpha_post_prec \ ...
                (alpha_like_prec*alpha_like_mean + alpha_prior_prec*alpha_prior_mean);
            % Drawing alphas here
            alpha_iter(i, :, p+1) = mvnrnd(alpha_post_mean, alpha_post_var);
            fit_ind(i, :, p+1) = xmat * alpha_iter(i, :, p+1)';
            ydev(i, :) = y_ind - fit_ind(i, :, p+1)';
        end
        
        % Drawing sigmasq from an inverse Gamma
        for k = 1:ngroups
            kk = find(ind_iter(:, p) == k);
            nk = length(kk);
            post_sig_a = (ntime-1) * nk/2 + prior_sig_a;
            post_sig_b = sum(sum(ydev(kk, :).^2)) / 2 + prior_sig_b;
            sigmasq_iter(k, p+1) = 1 ./ gamrnd(post_sig_a, 1/post_sig_b);

        % Drawing mu for the parametric part of random effects (What are the 
        % random effects here? The individual level slope and intercepts?)      
            if (nk == 0)
                post_mu_mean = prior_mu_mean;
                post_mu_var = inv(prior_mu_prec);
            else
                like_mu_mean = mean(alpha_iter(kk, :, p+1))';
                like_mu_prec = diag(nk ./ tau_alpha_iter(k, :, p));
                post_mu_prec = like_mu_prec + prior_mu_prec;
                post_mu_var = eye(ncovs + 1) / post_mu_prec;
                post_mu_mean = post_mu_var * ...
                    (prior_mu_prec*prior_mu_mean + like_mu_prec*like_mu_mean);
            end
            
            mu_alpha_iter(k, :, p+1) = mvnrnd(post_mu_mean, post_mu_var);

         % Drawing tau from a Gamma???
            tau_beta_a = nk/2 + prior_tau_a;
            tau_beta_b = sum(...
                            (alpha_iter(kk, :, p+1) - ...
                            (mu_alpha_iter(k, :, p+1)' * ...
                             ones(1, nk))').^2) + ...
                         prior_tau_b;
            u = rand(ncovs+1, 1);
            const1 = gamcdf(1 ./ prior_tau_uplim', ...
                            tau_beta_a * ones(ncovs+1, 1), ...
                            1 ./ tau_beta_b');
            const2 = ones(ncovs+1, 1) - u.*(ones(ncovs+1, 1) - const1);
            tau_alpha_iter(k, :, p+1) = 1./gaminv(const2, ...
                                                  ones(ncovs+1, 1) * tau_beta_a, ...
                                                  1./tau_beta_b');
        end
        
        % Evaluate the likelihood the data was drawn from the above distributions
        like = zeros(n, ngroups);
        for j = 1:ngroups % for each group (1 and 2)
            % Calculate the likelihood of each timeseries in each distribution

            like_group = prod(normpdf(ydev, ...
                                      zeros(n, ntime-1), ...
                                      sqrt(sigmasq_iter(j,p+1))*ones(n,ntime-1)),...
                                      2);
            % (like is likelihood score for each timeseries)
            %
            % numerator for Bayes rule:   
            numer(:, j) = like_group * prior_prob_iter(j, p);
        end

        % denominator for Bayes rule:
        denom = sum(numer, 2);

        % Calculate the posterior (Bayes rule)
        for j = 1:ngroups
            post_prob_iter(:, j, p+1) = numer(:, j) ./ denom;
        end

        % Update the group membership acccording to the posterior for each group
        u = rand(n, 1); % random threshold (why does this work???)
        kk = post_prob_iter(:, j, p+1) > u;
        ind_iter(:, p+1) = 1;
        ind_iter(kk, p+1) = 2;

        % Update the prior for each group <Slot in Davids code about here>
        prior_prob_iter(2, p+1) = betarnd(sum(kk), n-sum(kk));
        prior_prob_iter(1, p+1) = 1 - prior_prob_iter(2, p+1);

    end
end