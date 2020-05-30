# cluster-this
A short intro to clustering time series using MCMC  

## MCMC rationale

We want to know the group membership &gamma;<sub>k</sub> of each individual: where _Pr_(&gamma;<sub>k</sub> = _j_) is the probability that the individual _k_'s time series _y_<sub>k</sub> belongs to group _j_. Let's express this as the probability of observing _y_<sub>k</sub> given &gamma;<sub>k</sub>: Pr(_y_<sub>k</sub> | &gamma;<sub>k</sub> = _j_)  
  
Now we can begin with a simplified assumption that &gamma;<sub>k</sub> ~ ***N***(&mu;<sub>j</sub>, &sigma;<sub>j</sub><sup>2</sup>) 
...this simple starting assumption will be incrementally improved in time.

  
With these parameters, the likelihood function of observing _y_<sub>k</sub> is: 
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ***L***(_y_<sub>k</sub> | &mu;, &sigma;<sup>2</sup>, &gamma;<sub>k</sub> = _j_)

  
Where
  
&mu; = (&mu;<sub>1</sub>, &mu;<sub>2</sub>) is uniform and improper  
  
&sigma;<sup>2</sup> = (&sigma;<sub>1</sub><sup>2</sup>, &sigma;<sub>2</sub><sup>2</sup>)Pr(&sigma;<sub>j</sub><sup>2</sup>) &prop; 1/&sigma;<sup>2</sup> 
  
and &gamma;<sub>1</sub> ~ ***Be***(&pi;) where Pr(&gamma;<sub>k</sub> = 1) = &pi; and &pi; ~ ***U***[0, 1]
  
  
We want to explore the joint posterior (which is given by Bayes Rule):
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Pr(&gamma;, &mu;, &sigma;<sup>2</sup> | _y_) = Pr(y | &gamma;, &mu;, &sigma;<sup>2</sup>) Pr(&gamma;, &mu;, &sigma;<sup>2</sup>) / Pr(y)

  
Since this has no closed form solution, we must use MCMC to randomly sample &gamma;, &mu; and &sigma;<sup>2</sup> from Pr(&gamma;, &mu;, &sigma;<sup>2</sup> | _y_) and thus calculate the joint posterior.  
  
We can then update the probability of &gamma; according to ***Be***(&pi;).  
   
  
The posterior serves as the prior for the next iteration, adjusted by a random threshold. This is the Markov step in the MCMC scheme
