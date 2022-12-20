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

So far we have seen how an input vector$x ∈ \mathbb R^n$ can be fed to a layer of activation function units to create activations $a ∈ \mathbb R^m.$

**But what is theintuition behind doing so?**

> Let us consider the following `named entity recognition(NER)` problem in NLP as an example:
> 
> <div align=center>
> "Museums in Paris are amazing"
> </div>
> 
> Here, we want to classify whether or not the center word "Paris" is a named-entity.
> 
> In such cases, it is very likely that we would not just want to capture the presence of words in the window of word vectors but some other interactions between the words in order to make theclassifification. 
> 
> Such non-linear decisions can often not be captured by inputs fed directly to a Softmax function, but instead require the scoring of the intermediate layer.
> 
> We can thus use `another matrix` $\mathcal U ∈ \mathbb R^{m×1}$to generate an unnormalized score s for a classification task from the activations:
> 
> $$
> s = \mathcal U^Ta = \mathcal U^T f(Wx + b)
> $$
> 
> where f is the activation function.

### Maximum Margin Objective Function

> Like most machine learning models, such as `loss` function, neural networks also need an `optimization objective` as a measure of error or goodness which we want to minimize or maximize respectively. 

Here, we will discuss **a popular error metric** known as the `maximum margin objective`, which is most commonly associated with `Support Vector Machines(SVMs)`

The idea behind using this objective is very easy, just to:

**ensure that** `the score computed for "true" labeled data points` is higher than `the score computed for "false" labeled data points`. 

> Using the previous example, if we call the score computed for the "true" labeled window "Museums in Paris are amazing" as **s** 
> 
> and the score computed for the "false" labeled window "Not all museums in Paris" as $s_c$ (subscripted as c to signify that the window is "corrupt") 
> 
> Then, our objective function would be to:
> 
>         maximize $(s − s_c) $ or to minimize$ (s_c − s)$
> 
> However, we modify our objective to ensure that error is only computed if $s_c > s ⇒ (s_c − s) > 0$. The intuition behind doing this is that we `only care` the the "true" data point have a higher score than the "false" data point and that the rest does not matter. Thus, we want our error to be:
> $$
> (s_c − s)\quad if\,s_c > s,\quad else\, 0
> $$
> 
> Thus, our optimization objective is now:
> 
> $$
> minimize J = max(s_c − s, 0)
> $$
> 
> However, the above optimization objective is risky in the sense that it does not attempt to `create a margin of safety`. 
> 
> We would want the "true" labeled data point to score **higher than the "false" labeled data point by some positive margin ∆**. 
> 
> In other words, we would want error to be calculated if $(s − s_c < ∆)$ and not just when$(s − s_c < 0)$
> 
> Thus, we modify the optimization objective:
> 
> $$
> minimize J = max(∆ + s_c − s, 0)
> $$
> 
> In the above formulation 
> 
> $$
> s_c = \mathcal U^T f(Wx_c + b)\quad and\quad s = \mathcal U^T f(Wx + b)
> $$

___

### Training with Backpropagation – Elemental

> In this section we discuss how we train the different parameters in the model when the cost $J$ above is positive. No parameter updates are necessary if the cost is 0. 

Since we typically update parameters using gradient descent (or a variant such as SGD), we typically need the **gradient information** for any parameter as required in the update equation:

$$
θ^{(t+1)} = θ^{(t)} − α∇θ^{(t)}J

$$

`Backpropagation` is technique that allows us to use the `chain rule of differentiation to calculate loss gradients` for **any parameter** used in the feed-forward computation on the model. 

> To understand this further, let us understand the toy network shown in Figure below for which we will perform backpropagation. Here, we use a neural network with a single hidden layer and a single unit output.
> 
> <div align=center>
> <img title="" src="/assets/DLf5.png" alt="avatar" data-align="center">
> </div>

Let us establish some notation that will make iteasier to generalize this model later:

> - $x_i$ is an **input** to the neural network.
> 
> - $s$ is the **output** of the neural network.
>   
>   **layer**
> 
> - Each layer (including the input and output layers) has neurons which receive an input and produce an output. The $j$-th neuron of layer $k$ receives the **scalar input** $z^{(k)}_j$and produces the **scalar activation output** $a^{(k)}_j$.
> 
> - We will call the `backpropagated error` calculated at $z^{(k)}_j$as $\delta^{(k)}_j$
> 
> - Layer 1 refers to the input layer and not the first hidden layer. For the input layer, $x_j = z^{(1)}_j = a^{(1)}_j$.
> 
> - $W^{(k)}$ is the **transfer matrix** that maps the output from the $k$-th layer to the input to the$ (k + 1)$-th.
>   
>   Thus, $W^{(1)} = W$  and $W^{(2)} = \mathcal U$ to put this new generalized notation $\mathcal U$

**Let us begin:** 

Suppose the cost $J= (1 + s_c − s)$ is positive and we want to perform the update of parameter $W^{(1)}_{14}$ , we must realize that $W^{(1)}_{14}$ **only contributes to** $z^{(2)}_1$ and thus $a^{(2)}_1$. 

This fact is crucial to understanding backpropagation–backpropagated gradients are **only affected by values they contribute to**.



$a^{(2)}_1$ is consequently used in the forward computation of score by multiplication with $W^{(2)}_{1}$. We can see from the max-margin loss that:

$$
\frac{\partial J}{\partial s}=-\frac{\partial J}{\partial s_c}=-1
$$

Therefore we will work with $\frac{\partial s}{\partial W^{(1)}_{ij}}$ here for simplicity. 

Thus,


