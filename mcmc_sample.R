# setup
rm(list = ls())
# op = options()$warn
options(warn = 0)

n = 1000 # number of samples
nloop = 10000
y = matrix(NA_integer_, nrow=n, ncol=1) # y will hold our data (n timeseries)
prob_true = 0.5 # proportion of timeseries in group2 (mu2, sigma2)
u = runif(n)

# draw the samples for group1 from a normal distribution 
group1_idx = which(u > prob_true)
mu1_true = .125
sigma1_true = 0.1
y[group1_idx, ] = rnorm(length(group1_idx), mu1_true, sigma1_true)

# draw the samples for group2 from a normal distribution 
group2_idx = which(u < prob_true);
mu2_true = -.125;
sigma2_true = 0.10;
y[group2_idx, ] = rnorm(length(group2_idx), mu2_true, sigma2_true)

hist(y)

#### estimate group membership using Gibbs sampler ####
# setup (get starting values)
group_guess <- matrix(0, nrow = n, ncol = nloop+1)
u_guess <- runif(n)
kk = u_guess < 0.5 # find the index of each group member 
group_guess[kk, 1] = 2
group_guess[!kk, 1] = 1

mu_loop <- matrix(0, 2, nloop+1)
mu_loop[1, 1] = mean(y[kk, ])
mu_loop[2, 1] = mean(y[!kk, ])

sigmasq_loop = matrix(0, 2, nloop+1)
sigmasq_loop[1, 1] = var(y[kk, ])
sigmasq_loop[2, 1] = var(y[!kk, ])

prior_loop = matrix(0.5, 2, nloop+1)
posterior_loop = array(0.5, dim = c(n, 2, nloop+1))

for (p in 1:nloop) {
  
  # Draw mu and sigma for each group_j = 1, 2
  for (j in 1:2) {
    
    # Find the current group members
    kk = group_guess[, p] == j
    current_group = y[kk, ]
    
    # estimate the parameters of the group and then draw a random mu and sigma
    # from a distribution with the same parameters
    mu_loop[j, p+1] = rnorm(1, mean(current_group), 
                            sqrt(sigmasq_loop[j, p]/length(current_group))
                            )
    
    gamma_shape = length(current_group)/2
    gamma_rate = sum((current_group - mu_loop[j, p+1])^2)/2
    sigmasq_loop[j, p+1] = 1/rgamma(n=1, shape=gamma_shape, rate=gamma_rate)
  }
  
  # Evaluate the likelihood the data was drawn from a distribution with 
  # that new (randomly drawn) mu and sigma
  likelihood = matrix(0, n, 2)
  Bayes_numerator = matrix(0, n, 2)
  for (j in 1:2) {
    likelihood[, j] = dnorm(y, mu_loop[j, p+1], sqrt(sigmasq_loop[j, p+1]))
    likelihood[is.nan(likelihood[, j]), j] = 0 # fix for empty sets

    # Adjust likelihood by the prior (numerator in Bayes rule)
    Bayes_numerator[, j] = likelihood[, j] * prior_loop[j, p] # tbd: fix for numeric underflow
  }
  
  Bayes_denominator = rowSums(Bayes_numerator)  

  # Calculate the posterior (Bayes rule)
  for (j in 1:2) {
    posterior_loop[, j, p+1] = Bayes_numerator[, j] / Bayes_denominator
  }
  
  # Update the group membership according to the posterior + Gibbs
  Gibbs_threshold = runif(n) # random threshold
  kk = posterior_loop[, j, p+1] < Gibbs_threshold
  
  group_guess[kk, p+1] = 1
  group_guess[!kk, p+1] = 2

  # Update the prior for each group
  prior_loop[1, p+1] = rbeta(1, sum(kk), n - sum(kk))
  prior_loop[2, p+1] = 1 - prior_loop[1, p+1]
  
}

table(kk[group1_idx])
table(kk[group2_idx])

#### Results ####
matplot(t(mu_loop[, 1:10000]), type = "l", col = c("red", "blue"))
hist(mu_loop[1, 5000:nloop])
mean(mu_loop[1, 5000:nloop]) # ans: 0.4967641
mean(mu_loop[1, 5000:nloop]) - mu1_true # error: -0.00323585

hist(sigmasq_loop[1, 5000:nloop])
mean(sigmasq_loop[1, 5000:nloop]) # ans: 0.01008886
mean(sigmasq_loop[1, 5000:nloop]) - sigma1_true^2 # error: 8.885743e-05



guess1_idx = which(kk)
guess2_idx = which(!kk)

hist(y[group1_idx], xlim = c(-0.5, 0.5))
hist(y[group2_idx], xlim = c(-0.5, 0.5))


hist(y[which(!guess1_idx %in% group1_idx)], xlim = c(-0.5, 0.5))
hist(y[which(!guess2_idx %in% group2_idx)], xlim = c(-0.5, 0.5))




