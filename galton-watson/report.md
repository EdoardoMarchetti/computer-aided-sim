# Algorithm

The stopping criterion is based on the number of nodes at
generation i: if it is higher then a certain threshold (15), then we
assume that the tree will never be exhausted.

This is because the number of samples in a generation is strictly related
to the probability of extinction of the tree.
We have that the probability for a single node to not generate any child is k = P(Y=0), with 0<k<1.
Thus, to obtain the probability of generating 0 child with n samples, we need to compute k^n. As you can see, as n goes to infinity, k^n decrease exponentially, since k<1.

When n is greater or equal than 15, k^n is lower enough to consider the tree infinite.

The main algorithm is really simple and you can read it from the code in the class GWSimulator. The key point is that we extract a sample from a Poisson distribution for each node, in order to have the number of children of that node.
Therefore, at each generation, we extract this variable n times, where n is the number of nodes at the previous generation. Finally, we sum the extracted values, obtaining n+1, i.e., the number of samples in the current generation.

# Results

Looking at the obtained figures, we can state that our experiments are in line with the theory.
In particular, when increasing lambda, we obtain that the probability
of having an infinite tree increases. Also the number of experiments needed to obtain a tight confidence interval is higher.
