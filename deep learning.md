# Backprop and Neural Networks

## Foundations

> the need for non-linearclassifiers since most data are not linearly separable and thus, ourclassification performance on them is limited.

Neural networks area family of classifiers `with non-linear decision boundary`

### A Neuron

> - define    
>   
>   a generic computational unit that `takes n inputs` **x** and `produces a single output`, `scalar activation`
> 
> - weights
>   
>   What differentiates the outputs of different neurons is their `parameters` (also referred to as their `weights`), in form of an `n-dimensional weight vector` **w**
> 
> popular choices for neurons: "sigmoid" or "binary logistic regression" unit. 

This neuron is also associated with a `bias scalar` **b**. 

Input n-dimensional vector **x**, the output of this neuron is then:

$a =\frac{1}{1 + exp(−(w^Tx + b))}$

> both of w and x are column vector

We can also combine the weights and bias term above to equivalently formulate:

$a =\frac{1}{1 + exp(−[w^T\quad b] · [x\quad 1])}$

> the later 'b' and '1' are just following the relevant former, padding if necessary

### A Single Layer of Neurons


