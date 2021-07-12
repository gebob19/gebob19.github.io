---
title:  "First-Order Approximations of the Natural Gradient without the Tears"
description: "This blog post/tutorial dives deep into the theory and JAX code (similar to Pytorch and Tensorflow) for understanding the natural gradient and how to code first order approximations of the natural gradient"
layout: post
categories: journal
tags: [documentation,sample]
image: lion.jpg
mathjax: true
---

Suppose we have some loss function $$L$$ that we want to minimize and 
we have weights $$w$$. Throughout optimization we denote $$w^{(k)}$$ to the be the weights at the $$k$$-th iteration. 

Suppose normal optimization is noisy, has high variance, or for whatever reason we want to contrain how much our weights change between optimization iterations. This is called **proximal policy optimiztion**. 

We can then add another term to our loss function $$\rho(w^{(k)}, w^{(k+1)})$$ which minimizes how much the weights change between iterations like so: 

$$w^{(k+1)} = \text{prox}_{L, \lambda}(w^{(k)}) = \underset{w^{(k+1)}}{\argmin} [ L(w^{(k+1)}) + \lambda \rho(w^{(k)}, w^{(k+1)}) ]$$

where $$\lambda$$ controls how much the weights change between iterations.

## Euclidean Space

For example, in Euclidean space we could define $$\rho$$ to be the 2-norm:

$$\rho(w^{(k+1)}, w^{(k)}) = \dfrac{1}{2} ||w^{(k+1)} - w^{(k)}||^2$$)

And so in Euc space we get: 

$$w^{(k+1)} = \text{prox}_{L, \lambda}(w^{(k)}) = \underset{w^{(k+1)}}{\argmin} [ L(w^{(k+1)}) + \lambda \dfrac{1}{2} ||w^{(k+1)} - w^{(k)}||^2 ]$$

Lets try to solve for the optimum (i.e $$w^{(k+1)} = w^*$$) by computing the grad and setting to 0. If we do so, we get 

$$ 
\begin{align*}
0 &= \nabla L(w^*) + \lambda (w^* - w^{(k)}) \\
- \lambda^{-1} \nabla L(w^*) &= (w^* - w^{(k)}) \\
w^* &= w^{(k)} - \lambda^{-1} \nabla L(w^*) \\
\end{align*}
$$

Note that $$\nabla L$$ is defined at $$w^*$$ which we can't directly compute. But the equation looks very similar to SGD. You're probably wondering, "how could we solve something like this?"

To solve this there are a few approximations we could use...

## First Order Approximation of $$L$$ around $$w^{(k)}$$

Lets say we approximate $$L(w^{(k+1)})$$ with a first order taylor series around $$w^{(k)}$$:

$$
L(w^{(k+1)}) = L(w^{(k)}) + \nabla L(w^{(k)})^\intercal(w^{(k+1)} - w^{(k)})
$$

Then substituting this in to our loss we get:

$$\begin{align*}

\text{prox}_{L, \lambda}(w^{(k)}) =& \underset{w^{(k+1)}}{\argmin} [ L(w^{(k+1)}) + \lambda \rho(w^{(k+1)}, w^{(k)}) ] \\
\approx& \underset{w^{(k+1)}}{\argmin} [ L(w^{(k)}) + \nabla L(w^{(k)})^\intercal(w^{(k+1)} - w^{(k)}) + \lambda \rho(w^{(k+1)}, w^{(k)}) ] \\
=& \underset{w^{(k+1)}}{\argmin} [\nabla L(w^{(k)})^\intercal w^{(k+1)} + \lambda \rho(w^{(k+1)}, w^{(k)}) ] \\

\end{align*}$$

Note we go from the second to the third line since $$\underset{w^{(k+1)}}{\argmin}$$ ignores any terms without $$w^{(k+1)}$$. 

Computing the gradient wtr to $$w^{(k+1)}$$ and setting it to zero again we get 

$$ 
\begin{align*}
0 &= \nabla L(w^{(k)}) + \lambda (w^* - w^{(k)}) \\
w^* &= w^{(k)} - \lambda^{-1} \nabla L(w^{(k)}) \\
\end{align*}
$$

This is the gradient descent equation. Then in Euclidean space, proximal policy optimiztion is the same as regular gradient descent. Interesting! 

## Use a Second Order Approximation of $$\rho$$

Another way we could solve the equation is to use a second order approximation of $$\rho$$ and let $$\lambda \rightarrow \infty$$ (we want our steps to be as close as possible). 

*Note:* Though we don't need to do this since we solved it directly in the last section, the ideas in this simple example will complement the rest of the post nicely. 

Since $$\lambda \rightarrow \infty$$, in Euclidean space, $$w^{(k+1)} \approx w^{(k)}$$ and so $$\rho (w^{(k+1)}, w^{(k)}) = 0$$ and $$\nabla \rho (w^{(k+1)}, w^{(k)}) = 0$$

$$\rho (w^{(k+1)}, w^{(k)}) = \dfrac{1}{2} ||w^{(k)} - w^{(k+1)} ||^2 \approx \dfrac{1}{2} ||w^{(k)} - w^{(k)} ||^2 = 0$$ 

and 

$$\nabla \rho (w^{(k+1)}, w^{(k)}) = (w^{(k)} - w^{(k+1)}) \approx (w^{(k)} - w^{(k)}) = 0$$

Then when we approx $$\rho$$ with a second-order taylor series, we are left with only our second order approximation: 

$$\begin{align*}
\rho (w^{(k+1)}, w^{(k)}) &\approx \rho(w^{(k+1)}, w^{(k)}) + \nabla \rho(w^{(k+1)}, w^{(k)})^\intercal (w^{(k+1)} - w^{(k)}) + \dfrac{1}{2} (w^{(k+1)} - w^{(k)})^\intercal G (w^{(k+1)} - w^{(k)}) \\
&= \dfrac{1}{2} (w^{(k+1)} - w^{(k)})^\intercal G (w^{(k+1)} - w^{(k)})
\end{align*}$$

where $$ G = \nabla^2 \rho (w^{(k+1)}, w^{(k)}) |_{w^{(k+1)} = w^{(k)}} $$. 

Using the second-order approx of $$\rho$$ with the first-order approx of $$L$$ which we derived last section we get the following: 

$$\begin{align*}
\text{prox}_{L, \lambda}(w^{(k)}) =& \underset{w^{(k+1)}}{\argmin} [ L(w^{(k+1)}) + \lambda \rho(w^{(k)}, w^{(k+1)}) ] \\
\approx& \underset{w^{(k+1)}}{\argmin} [\nabla L(w^{(k)})^\intercal w^{(k+1)} + \lambda \dfrac{1}{2} (w^{(k+1)} - w^{(k)})^\intercal G (w^{(k+1)} - w^{(k)})]
\end{align*}$$

Computing the gradient and setting it = 0 we get:

$$\begin{align*}
w^* &= w^{(k)} - \lambda^{-1} \text{G}^{-1} \nabla L(w^{(k)}) 
\end{align*}$$

In Euclidean space $$\text{G} = \nabla^2 \rho (w^{(k+1)}, w^{(k)}) = \text{I}$$, so again we get gradient descent. 

$$\begin{align*}
w^* &= w^{(k)} - \lambda^{-1} \text{G}^{-1} \nabla L(w^{(k)}) \\
&= w^{(k)} - \lambda^{-1} \text{I}^{-1} \nabla L(w^{(k)}) \\
&= w^{(k)} - \lambda^{-1} \nabla L(w^{(k)}) 
\end{align*}$$

Though this doesn't conclude anything new, this simple example shows that we can use a second-order approximation for $$\rho$$ and derive an accurate update rule. 

## KL Divergence

A problem which occus is that Euclidean space isn't always great (taken from [here](https://wiseodd.github.io/about/)): 

![](param_space_dist.png)
![](param_space_dist2.png)

In this example, both images, the Euclidean distance (red line), or the **parameter space**, of the Gaussians are the same (i.e 4). However, we see that the distrbutions should not have the same distance metric (the top image should have a smaller distance than the bottom image). KL-Divergence accounts for this and measures the distance in **distribution space**.

Specifically, KL-Divergence measures the distance between two distributions $$p(x | \theta')$$ and $$p(x | \theta)$$ and is defined as: 

$$\begin{align*}
D_{KL}[p(x | \theta) || p(x | \theta')] &= \sum_x p(x|\theta) [\log \dfrac{p(x | \theta)}{p(x | \theta')}] \\
&= \sum_x p(x|\theta) [\log p(x | \theta) - \log p(x | \theta')] \\
&= \mathbb{E}_{p(x | \theta)} [\log p(x | \theta) - \log p(x | \theta')] \\
\end{align*}$$

KL measures the distance between distributions so it cant directly be applyed to neural nets with weights $$w$$. To show this difference we let $$\theta^{(k)}$$ parameterize some distribution $$p(x | \theta^{(k)})$$ at the $$k$$-th step of optimization. 

*Teaser:* We'll see how to apply this to neural nets later. 

Similar to before, lets assume we want $$\theta^{(k)}$$ and $$\theta^{(k+1)}$$ to be close across steps. Then we can let $$\rho(\theta^{(k)}, \theta^{(k+1)}) = D_{KL}(p_{\theta^{(k)}} || p_{\theta^{(k+1)}})$$ with the shorthand $$p_{\theta^{(k)}} = p(x | \theta^{(k)})$$.

Now we want to follow a similar procedure from the last section and solve 

$$\begin{align*}
\text{prox}_{L, \lambda}(\theta^{(k)}) =& \underset{\theta^{(k+1)}}{\argmin} [ L(\theta^{(k+1)}) + \lambda \rho(\theta^{(k)}, \theta^{(k+1)}) ] \\ 
=& \underset{\theta^{(k+1)}}{\argmin} [ L(\theta^{(k+1)}) + \lambda D_{KL}(p_{\theta^{(k)}} || p_{\theta^{(k+1)}}) ] \\ 
\end{align*}$$

Similar to last section, lets approximate $$D_{KL}(p_{\theta^{(k)}} || p_{\theta^{(k+1)}})$$ with a second-order taylor expansion.

Recall from the previous section to approximate $$\rho$$ with a second-order taylor series we need to define $$\rho|_{\theta'=\theta}$$, $$\nabla \rho|_{\theta'=\theta}$$ and $$\nabla^2 \rho|_{\theta'=\theta}$$.  

First, lets derive $$\rho$$, $$\nabla \rho$$, and $$\nabla^2 \rho$$ for $$D_{KL}$$

$$\begin{align*}
\rho = D_{KL}[p(x | \theta) || p(x | \theta')] &= \mathbb{E}_{p(x | \theta)} [\log p(x | \theta) - \log p(x | \theta')] \\
&= \mathbb{E}_{p(x | \theta)} [\log p(x | \theta)] - \mathbb{E}_{p(x | \theta)}[\log p(x | \theta')] \\
\end{align*}$$

...

$$\begin{align*}

\nabla \rho = \nabla_{\theta'} D_{KL}[p(x | \theta) || p(x | \theta')] &= \nabla_{\theta'} \mathbb{E}_{p(x | \theta)} [\log p(x | \theta)] - \nabla_{\theta'} \mathbb{E}_{p(x | \theta)}[\log p(x | \theta')] \\
&= - \nabla_{\theta'} \mathbb{E}_{p(x | \theta)}[\log p(x | \theta')] \\
&= - \nabla_{\theta'} \int p(x | \theta) \log p(x | \theta') \text{d}x \\
&= - \int p(x | \theta) \nabla_{\theta'} \log p(x | \theta') \text{d}x

\end{align*}$$

...

$$\begin{align*}
\nabla^2 \rho = \nabla_{\theta'}^2 D_{KL}[p(x | \theta) || p(x | \theta')] &= - \int p(x | \theta) \nabla_{\theta'}^2 \log p(x | \theta') \text{d}x \\
\end{align*}$$

To evaluate $$\rho|_{\theta'=\theta}$$, $$\nabla \rho|_{\theta'=\theta}$$ and $$\nabla^2 \rho|_{\theta'=\theta}$$ we are going to need the two following equations: 

$$\mathbb{E}_{p(x|\theta)} [\nabla_{\theta} \log p(x | \theta)] = 0$$
and 

$$\mathbb{E}_{p(x|\theta)} [ \nabla_{\theta}^2 \log p(x | \theta) ] = -\text{F}$$

Where $$\text{F} = \mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ \nabla \log p(x \vert \theta) \, \nabla \log p(x \vert \theta)^{\text{T}} \right]$$ is the fisher information matrix. For the full derivation of these eqns checkout this blog post: 

https://wiseodd.github.io/techblog/2018/03/11/fisher-information/

And so using these equations, we get: 

$$\begin{align*}

\nabla_{\theta'} D_{KL}[p(x | \theta) || p(x | \theta')] |_{\theta' = \theta} &= - \int p(x | \theta) \nabla_{\theta'} \log p(x | \theta')|_{\theta' = \theta} \text{d}x \\
&= - \mathbb{E}_{p(x|\theta)} [\nabla_{\theta} \log p(x | \theta)] \\
&= 0
\end{align*}$$

...

$$\begin{align*}
\nabla_{\theta'}^2 D_{KL}[p(x | \theta) || p(x | \theta')] |_{\theta' = \theta} &= - \int p(x | \theta) \nabla_{\theta'}^2 \log p(x | \theta') |_{\theta' = \theta} \text{d}x \\
&= - \mathbb{E}_{p(x|\theta)} [ \nabla_{\theta}^2 \log p(x | \theta) ] \\
&= \text{F} \\
\end{align*}$$

Then using the first-order approximation of $$L$$ with our second-order approximation of $$D_{KL}$$ we get:

$$\begin{align*}
\text{prox}_{L, \lambda}(\theta^{(k)}) =& \underset{\theta^{(k+1)}}{\argmin} [ L(\theta^{(k+1)}) + \lambda \rho(\theta^{(k)}, \theta^{(k+1)}) ] \\ 
=& \underset{\theta^{(k+1)}}{\argmin} [ L(\theta^{(k+1)}) + \lambda D_{KL}(p_{\theta^{(k)}} || p_{\theta^{(k+1)}}) ] \\ 
\approx& \underset{\theta^{(k+1)}}{\argmin} [\nabla L(\theta^{(k)})^\intercal \theta^{(k+1)} + \lambda \dfrac{1}{2} (\theta^{(k+1)} - \theta^{(k)})^\intercal F (\theta^{(k+1)} - \theta^{(k)})] \\
\end{align*}$$

Computing the gradient and setting it to zero we get 

$$\begin{align*}
\theta^* &= \theta^{(k)} - \lambda^{-1}\text{F}^{-1}\nabla L(\theta^k) \\
&= \theta^{(k)} - \lambda^{-1}\tilde{\nabla} L(\theta^k) \\
\end{align*}$$

Where $$\tilde{\nabla} L(\theta^k)$$ is called the **natural gradient**. And this update rule is called **natural gradient descent**. 

But how do we apply it to neural networks? 

# Natural Gradient Descent for Neural Nets 

Usually our networks output distributions $$r( \cdot | x)$$ which we can use (e.g multi-classification problem). 

The idea is instead of making sure the weights don't change, we can make sure the output distribution doesn't change a lot. 

So we can define our distribution as $$z = f(w, x)$$ with $$\rho = D_{KL}$$ as the change over our data distribution. We define $$p_{\text{pull}}$$ as the following:

$$
\rho_{\text{pull}}(w, w') = \mathbb{E}_{x}[f_x * \rho(w, w')] = \mathbb{E}[\rho(f(w, x), f(w', x))]
$$

Where the $$f_x*\rho$$ operator is called a pullback operation: 

$$
f*g(x_1, ..., x_k) = g(f(x_1), f(x_2), ..., f(x_k))
$$

Now similar to the previous sections, lets approximate $$p_{\text{pull}}$$ with a second-order taylor expansion. Again, we need to find $$\rho_{\text{pull}}(w, w')$$, $$\nabla \rho_{\text{pull}}(w, w')$$, and $$\nabla_{\text{pull}}^2 \rho(w, w')$$

$$\begin{align*}
\rho_{\text{pull}}(w, w')|_{w = w'} &= \rho(f(w', x), f(w', x)) \\
&= 0 \\
\end{align*}$$

$$\begin{align*}
\nabla_w \rho_{\text{pull}}(w, w')|_{w = w'} &= \nabla_w\rho(f(w, x), f(w', x)) \\
&= \text{J}_{zx}^\intercal \underbrace{\nabla_z\rho(z, z')|_{z=z'}}_{=0} \\ 
&= 0 \\
\end{align*}$$

Recall the last line occurs since we know for $$\nabla \rho = \nabla D_{KL} = 0$$.

*Note:* Notice how pullback functions ($$\rho_{\text{pull}}$$) make the gradients simple to compute.

Using this decomposition it can be shown that the hessian can also be decomposed into the following (check out this for derivation: https://andrew.gibiansky.com/blog/machine-learning/gauss-newton-matrix/): 

$$\begin{align*}
\nabla^2 L_{x, t}(w) &= \text{J}_{zw}^\intercal H_z \text{J}_{zw} + \underbrace{\sum_a \overbrace{\dfrac{d \rho}{dz_a}}^{=0} \nabla^2_w [f(x, w)]_a}_{=0} \\
&= \text{J}_{zw}^\intercal H_z \text{J}_{zw}
\end{align*}$$

*Note:* There are many connections to the Guass-Newton decomposition of the Hessian. 

Again, the second term disappears since we know $$\nabla \rho = \nabla D_{KL} = 0$$.

For the case of $$D_{KL}$$ we know that $$H_z = F_z$$ where $$F_z$$ is the fisher information matrix. And so our second-order approximation of the neural network is: 

$$ \begin{align*}
F_w &= \mathbb{E_{x \sim p_{\text{data}}}} [\text{J}_{zw}^\intercal F_z \text{J}_{zw}] \\
&= \mathbb{E_{x \sim p_{\text{data}}}} [\text{J}_{zw}^\intercal \mathbb{E}_{t \sim r(\cdot | x)} [\nabla_z \log r(t | x) \nabla_z \log r(t | x) ^ \intercal] \text{J}_{zw}] \\
&= \mathbb{E_{x \sim p_{\text{data}}, {t \sim r(\cdot | x)}}} [\nabla_w \log r(t | x) \nabla_w \log r(t | x) ^ \intercal] \\
\end{align*}$$

Notice $${t \sim r(\cdot | x)}$$ samples $$t$$ from the model's distribution which is the **true fisher information matrix**. 

Similar, but not the same, is the **empirical fisher information matrix** where $$t$$ is the data's target. 

In total, using our approximation we get the following second-order approximation: 

$$
\rho_{\text{pull}}(w, w') \approx \dfrac{1}{2} (w - w')^\intercal \text{F}_w (w - w')
$$

Which again, if we compute the gradient and set it to zero, we get: 

$$\begin{align*}
w^{(k+1)} &= \underset{w^{(k+1)}}{\argmin} [\nabla L(w^{(k)})^\intercal w^{(k+1)} + \lambda \dfrac{1}{2} (w^{(k)} - w^{(k+1)})^\intercal \text{F}_w (w^{(k)} - w^{(k+1)})] \\

0 &= \nabla L(w^{(k)}) + \lambda (w^{(k)} - w^*)^\intercal \text{F}_w  \\

- \lambda^{-1} \text{F}_w^{-1}\nabla L(w^{(k)}) &=  (w^{(k)} - w^*)   \\
w^* &= w^{(k)} - \lambda^{-1} \text{F}_w^{-1}\nabla L(w^{(k)})   \\

\end{align*}$$

Which results in the natural gradient method. 

# Computing the Natural Gradient in JAX 

We'll setup some code and then train a small model on MNIST dataset.

## Code

First, we define a small MLP model 

<script src="https://gist.github.com/gebob19/d0b4e9ef147545b3f5762461c4b2a6fe.js"></script>

Then we define a few loss functions 

<script src="https://gist.github.com/gebob19/e288e7a0646d53ee732b6ab9c251b705.js"></script>

Next, we make the losses work for batch input using `vmap`

<script src="https://gist.github.com/gebob19/c2ad9cf658c9fb4a2377614c7dbf8a80.js"></script>

To get a feel for JAX, we first define a normal gradient step function 

<script src="https://gist.github.com/gebob19/d519682cfe29a73f71cdecfc3eb2566b.js"></script>

We can then define a natural gradient step using the empirical fisher matrix like so:

<script src="https://gist.github.com/gebob19/2ecb5df93f8f2e7ebb58c0823084bfa6.js"></script>

*Note:* we can use the `grads` from the `mean_cross_entropy` instead of the `mean_log_likelihood` to improve the code

*Exercise:* Why? (Hint: `grad2fisher`)

Finally, we can define the natural gradient step by sampling from the model's predictions: 

<script src="https://gist.github.com/gebob19/ba7d73eb16c8331cc75a7b5dd49e7ab9.js"></script>

Tada! Done! Last but not least we define some boilerplate training code and compare the runs: 

<script src="https://gist.github.com/gebob19/7200ee0bfc99274b817896cbade5dd8c.js"></script>

## Results 

*Note:* In the code, the gradients actually blow up. I think this is because our $$\lambda \rightarrow \infty$$ doesn't hold in practice and so $$\lambda^{-1} \text{F}_w^{-1}$$ results in very large numbers. To get around this I use gradient clipping. 

Here are the tensorboard results: 

### Test Accuracy
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/assets/natural_grad/testacc.png" alt="test-acc" class="center"/>
</div>

### Train Loss
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/assets/natural_grad/loss.png" alt="train-loss" class="center"/>
</div>

### Linear 0
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/assets/natural_grad/w0.png" class="center"/>
</div>
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/assets/natural_grad/b0.png" class="center"/>
</div>

### Linear 1
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/assets/natural_grad/w2.png" class="center"/>
</div>
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/assets/natural_grad/b2.png" class="center"/>
</div>

Unfortantely in this scenario, I found Natural Gradient desecent approximations don't outperform SGD :(
































