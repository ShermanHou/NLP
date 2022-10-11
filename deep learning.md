# Backprop and Neural Networks

## Foundations

> the need for non-linearclassifiers since most data are not linearly separable and thus, ourclassification performance on them is limited.

Neural networks area family of classifiers `with non-linear decision boundary`

____

### A Neuron

> - define    
>   
>   a generic computational unit that `takes n inputs` **x** and `produces a single output`, `scalar activation`.
> 
> - weights
>   
>   What differentiates the outputs of different neurons is their `parameters` (also referred to as their `weights`), in form of an `n-dimensional weight vector` **w**
> 
> popular choices for neurons: "sigmoid" or "binary logistic regression" unit. 

This neuron is also associated with a `bias scalar` **b**. 

Input n-dimensional vector **x**, the output of this neuron is then:

$$
a =\frac{1}{1 + exp(−(w^Tx + b))}
$$

> both of w and x are column vector

We can also combine the weights and bias term above to equivalently formulate:

$$
a =\frac{1}{1 + exp(−[w^T\quad b] · [x\quad 1])}
$$

> the later 'b' and '1' are just following the relevant former, padding if necessary
> 
> the following figure shows all the things.

<div align=center>
<img title="" src="/assets/DLf2.png" alt="avatar" data-align="center">
</div>

____

### A Single Layer of Neurons

> We extend the idea above to `multiple neurons`, the case where the input **x** is fed as an common input to them.

<div align=center>
<img title="" src="/assets/DLf3.png" alt="avatar" data-align="center">
</div>

If we refer to the different neurons’ weights as $\{w^{(1)}, · · · , w^{(m)}\}$and the biases as ${b_1,···,b_m}$, we can say the respective activations are $\{a_1, ··· , a_m\}$:

$$
a_i=\frac{1}{1+exp(w^{(i)T}x+b_i)}
$$

> each weights vector and biases constitute a units set of a neuron

____

Let us define the following abstractions to keep the notation simpleand useful for more complex networks:

$$
\sigma(z)=\begin{bmatrix}
\frac{1}{1+exp(z_1)}\\
\vdots
\\
\frac{1}{1+exp(z_m)}
\end{bmatrix}
$$

$$
b=\begin{bmatrix}
b_1\\
\vdots
\\
b_m
\end{bmatrix}\in \mathbb{R}^m
$$

$$
W=\begin{bmatrix}
-\quad w^{(1)T}\quad-\\
\vdots
\\
-\quad w^{(m)T}\quad-\\
\end{bmatrix}\in \mathbb{R}^{m\times n}
$$

We can now write the output of scaling and biases as:

$$
z = Wx+b
$$

The activations of the sigmoid function can then be written as:

$$
\begin{bmatrix}
a^{(1)}\\
\vdots
\\
a^{(m)}
\end{bmatrix}=\sigma (z)=\sigma (Wx+b)
$$

So what do these activations really tell us? 

Well, one can think of these activations as `indicators of the presence` of some `weighted combination of features`. 

We can then use **a combination of these activations** to perform **classification** tasks.

____

### Feed-forward Computation
