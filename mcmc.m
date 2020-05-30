function [ind_iter] = mcmc(y)
    % Initializing values for MCMC
    n = size(y, 1); % update n
    nloop = 10000; % number of iterations
    nwarmup = 5000;
    ind_iter = ones(n, nloop+1); % index for group membership in each loop
    u = rand(n, 1); % random starting value for proportion assumed in group2
    kk = u < 0.5; % find the index of each group member 
    ind_iter(kk, 1) = 2; % update the first index with starting group

    mu_iter = zeros(2, nloop+1); % to store the mu in each loop 
    mu_iter(1, 1) = mean(mean(y(kk, :), 2)); % mu of group1 to start
    mu_iter(2, 1) = mean(mean(y(~kk, :), 2)); % mu of group2 to start

    sigmasq_iter = zeros(2, nloop+1); % to store sigmasq in each loop
    temp = y(kk, :);
    sigmasq_iter(1, 1) = var(temp(:));
    temp = y(~kk, :);
    sigmasq_iter(2, 1) = var(temp(:));

    prior_prob_iter = 0.5 * ones(2, nloop+1); % priors
    post_prob_iter = 0.5 * ones(n, 2, nloop+1); % posteriors

    % Starting MCMC scheme
    for p = 1:nloop

        % Drawing mu and sigma
        for j = 1:2 % for each group (1 and 2)

            % find the current group scores
            kk = ind_iter(:,p) == j;
            temp = y(kk, :);
            temp = temp(:);

            % draw a random mu and sigma from the same distribution 
            mu_iter(j, p+1) = normrnd(mean(temp), ...
                sqrt(sigmasq_iter(j,p) / length(temp)));
            sig_a = length(temp) / 2; % shape and scale parameters for gamma
            sig_b = sum((temp-ones(length(temp), 1) * mu_iter(j, p+1)).^2) / 2;
            sigmasq_iter(j, p+1) = 1 ./ gamrnd(sig_a, 1/sig_b);
        end

        % Evaluate the likelihood the data was drawn from a distribution with 
        % that new (random) mu and sigma
        like = zeros(n, 2);
        for j = 1:2 % for each group (1 and 2)
            % Calculate the likelihood of each timeseries in each distribution
            like(:,j) = prod(...
                             normpdf(...
                                     y(:,:), ...
                                     mu_iter(j, p+1), ... % new mu
                                     sqrt(sigmasq_iter(j, p+1))... % new sigma
                                     ),...
                             2); % take the product of each row (timeseries)
            % (like is likelihood score for each timeseries)
            %
            % numerator for Bayes rule:   
            numer(:,j) = like(:, j) .* ones(n, 1) * prior_prob_iter(j, p);
        end

        % denominator for Bayes rule:
        denom = sum(numer, 2);

        % Calculate the posterior (Bayes rule)
        for j=1:2
            post_prob_iter(:, j, p+1) = numer(:, j) ./ denom;
        end

        % Update the group membership acccording to the posterior for each group
        u = rand(n, 1); % random threshold (why does this work???)
        kk = post_prob_iter(:, j, p+1) < u;
        ind_iter(kk, p+1) = 1;
        ind_iter(~kk, p+1) = 2;

        % Update the prior for each group <Slot in Davids code about here>
        prior_prob_iter(1, p+1) = betarnd(sum(kk), n-sum(kk));
        prior_prob_iter(2, p+1) = 1 - prior_prob_iter(1, p+1);

    end
end