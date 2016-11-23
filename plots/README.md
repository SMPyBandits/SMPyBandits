# Some illustrations for [this project](https://github.com/Naereen/AlgoBandits)

Here are some plots illustrating the performances of the different [policies](../Policies/) implemented in this project, against various problems (with [`Bernoulli`](../Arms/Bernoulli.py) arms only):

### Small tests
[![5 tests - AdBandit and Aggr](5_tests_AdBandit__et_Aggr.png)](5_tests_AdBandit__et_Aggr.png)
[![2000 steps - 100 repetition](2000_steps__100_average.png)](2000_steps__100_average.png)

### Larger tests
[![10000 steps - 50 repetition - 6 policies - With 4 Aggr](10000_steps__50_repetition_6_policies_4_Aggr.png)](10000_steps__50_repetition_6_policies_4_Aggr.png)
[![10000 steps - 50 repetition - 6 policies - With Softmax and 1 Aggr](10000_steps__50_repetition_6_policies_with_Softmax_1_Aggr.png)](10000_steps__50_repetition_6_policies_with_Softmax_1_Aggr.png)

### Some examples where [`Aggr`](../Policies/Aggr.py) performs well
[![Aggr is the best here](Aggr_is_the_best_here.png)](Aggr_is_the_best_here.png)
[![one Aggr does very well](one_Aggr_does_very_well.png)](one_Aggr_does_very_well.png)

### One last example
The [`Aggr`](Policies/Aggr.py) can have a fixed learning rate, whose value has a great effect on its performance, as illustrated here:
[![20000 steps - 100 repetition - 6 policies - With 5 Aggr](20000_steps__100_repetition_6_policies_5_Aggr.png)](20000_steps__100_repetition_6_policies_5_Aggr.png)

### One a harder problem
[![example harder problem](example_harder_problem.png)](example_harder_problem.png)
