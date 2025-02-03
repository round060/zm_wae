data {
  int<lower=0> N;
  int<lower=0> n_year;
  int<lower=0> n_lake;
  int<lower=0> n_post;
  int<lower=0> n_trt;
  // observation-level predictors
  array[N] int<lower=0, upper=1> post;
  array[N] int<lower=0, upper=1> trt;
  vector[N] fry_pa;
  vector[N] fgl_pa;
  vector[N] survey_gdd_sum;
  vector[N] offsets;
  // lake and year indicator
  array[N] int<lower=0, upper=n_lake> lake;
  array[N] int<lower=0, upper=n_year> year;
  // response variable
  array[N] int<lower=0> y;
  // lake-level predictors
  vector[n_lake] gdd;
  vector[n_lake] lake_area;
}
parameters {
  real<lower=0> sigma_year;
  real<lower=0> sigma_lake;
  
  real<lower=0, upper=1> theta;
  
  real b_0;
  real b_post;
  real b_trt;
  real b_fry;
  real b_fgl;
  real b_post_trt;
  real b_survey_gdd_sum;

  real b_lake_area;
  real b_gdd;

  vector[n_year] b_year;
  
  vector[n_lake] b_hat;
}
model {
  vector[N] lambda;
  //vector[N] theta;                      // probability of zero 
  vector[n_lake] b_lake_hat;
  b_0 ~ normal(0, 5);
  b_post ~ normal(0, 5);
  b_trt ~ normal(0, 5);
  b_fry ~ normal(0, 5);
  b_fgl ~ normal(0, 5);
  b_post_trt ~ normal(0, 5);
  b_survey_gdd_sum ~ normal(0, 5);
  
  b_year ~ normal(0, sigma_year);
  
  b_lake_area ~ normal(0,5);
  b_gdd ~ normal(0, 5);

  for (j in 1:n_lake)
    b_lake_hat[j] = b_lake_area*lake_area[j] + b_gdd*gdd[j];
  
    b_hat ~ normal(b_lake_hat, sigma_lake);
  
  
  for (i in 1:N){
    lambda[i] = offsets[i] + b_hat[lake[i]] + b_year[year[i]] + b_0
    + b_post*post[i] + b_trt*trt[i] + b_post_trt*post[i]*trt[i] 
    + b_fry*fry_pa[i] + b_fgl*fgl_pa[i] + 
    + b_survey_gdd_sum*survey_gdd_sum[i];
    
  }
  
  // likelihood
  for (n in 1:N) {
    if (y[n] == 0)
      target += log_sum_exp(bernoulli_lpmf(1 | theta),
                            bernoulli_lpmf(0 | theta)
                            + poisson_lpmf(y[n] | exp(lambda[n])));
      else
        target+= bernoulli_lpmf(0 | theta)
        + poisson_lpmf(y[n] | exp(lambda[n]));
  }
}
