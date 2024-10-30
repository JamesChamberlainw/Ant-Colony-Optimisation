# Ant Colony Optimisation
ECMM409: Ant Colony Optimisation for BankProblem dataset provided by Exeter University. 

The problem: using Ant Colony Optimisation (ACO), optimise a bank van so that it will carry the most money for its maximum weight of 295kg. 

## Construction of the Heuristic Matrix 

A heuristic needs to be selected, as either weight or value alone will not do, as it leaves out crucial information on the other half. Combining them is the best solution, given that no other data can outline their relationship. 

The head of the data looks like this: 
```py
   weight  value
0     9.4   57.0
1     7.4   94.0
2     7.7   59.0
3     7.4   83.0
4     2.9   82.0
``` 
(plus one to index to get bag number)

The selected huristic will be value per weight

$$
    value\ per\ weight = \frac{weight}{value}
$$

$$
    vpw = \frac{weight}{value} 
$$

Thus vpw can now be used to state the value of the weight being added to the van - this can now be used to generate the distance matrix what can then be used to produce the heuristic matrix.  

```py
[[0.         0.13759298 0.07551697 ... 0.0834917  0.10746499 0.16746493]
 [0.0558301  0.         0.07551697 ... 0.0834917  0.10746499 0.16746493]
 [0.0558301  0.13759298 0.         ... 0.0834917  0.10746499 0.16746493]
 ...
 [0.0558301  0.13759298 0.07551697 ... 0.         0.10746499 0.16746493]
 [0.0558301  0.13759298 0.07551697 ... 0.0834917  0.         0.16746493]
 [0.0558301  0.13759298 0.07551697 ... 0.0834917  0.10746499 0.        ]]
 ```

As you may have already noticed, the vertical columns are all the same. This is due to the fact that they are not related to each other; we are looking for the next best `vpw`. However, there is still some learning to be done, as the ACO needs to find what ones fit in for 295kg. 