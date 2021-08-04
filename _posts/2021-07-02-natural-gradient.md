---
title:  "Natural Gradient Descent without the Tears"
description: "This blog post/tutorial dives deep into the theory and JAX code (similar to Pytorch and Tensorflow) for understanding the natural gradient and how to code approximations of the natural gradient"
layout: post
categories: journal
tags: [documentation,sample]
image: nasa.jpeg
mathjax: true
---

[Photo Link](https://unsplash.com/photos/6-jTZysYY_U)

To set the scene, suppose we have a model with weights $$w$$ and some loss function $$L(w)$$ that we want to minimize. Then our objective is to find $$w^*$$ where: 

$$w^* = \underset{w}{\text{argmin}} L(w)$$

Suppose good ol' SGD optimization creates noisy gradients or has a high variance (e.g policy gradient methods in RL). One thing we could do to smooth out the optimization procedure is minimize how much our weights change between optimization iterations. This is called a **proximal optimization method**. 

To do this, with $$w^{(k)}$$ denoting the weights at the $$k$$-th iteration, we can add another term to our loss function $$\rho(w^{(k)}, w^{(k+1)})$$ which minimizes how much the weights change between iterations like so: 

$$w^{(k+1)} = \text{prox}_{L, \lambda}(w^{(k)}) = \underset{w^{(k+1)}}{\text{argmin}} [ L(w^{(k+1)}) + \lambda \rho(w^{(k)}, w^{(k+1)}) ]$$

where $$\lambda$$ controls how much we want the weights to change between iterations (larger $$\lambda$$ = less change).

***

Most of this post is created in reference to Roger Grosse's fantastic 'Neural Net Training Dynamics' course. The course can be found [here](https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2021/) and I would highly recommend checking it out. Though there are no lecture recordings, the course notes are in a league of thier own.

***

# Euclidean Space

For example, in Euclidean space we could define $$\rho$$ to be the 2-norm between our weights:

$$\rho(w^{(k+1)}, w^{(k)}) = \dfrac{1}{2} \| w^{(k+1)} - w^{(k)}\|^2$$

And so we would get: 

$$w^{(k+1)} = \text{prox}_{L, \lambda}(w^{(k)}) = \underset{w^{(k+1)}}{\text{argmin}} [ L(w^{(k+1)}) + \lambda \dfrac{1}{2} \|w^{(k+1)} - w^{(k)}\|^2 ]$$

Lets try to solve for the optimum (i.e $$w^{(k+1)} = w^*$$) by computing the grad and setting to 0. If we do so, we get:

$$ 
\begin{align*}
0 &= \nabla L(w^*) + \lambda (w^* - w^{(k)}) \\
- \lambda^{-1} \nabla L(w^*) &= (w^* - w^{(k)}) \\
w^* &= w^{(k)} - \lambda^{-1} \nabla L(w^*) \\
\end{align*}
$$

Note that $$\nabla L$$ is defined at $$w^*$$ so we can't directly use the result. But the equation looks very similar to SGD. Naturally, you're probably wondering: 

> "Brennan, why did you introduce a result which we can't even use? This is blasphemy."

Well fear not my fellow gradient friends, to solve this there are a few approximations we could use...

# First Order Approximation of $$L$$ around $$w^{(k)}$$

**Note**: For the following sections it would be good to be comfortable with Taylor series approximations. If you aren't, I would recommend grabbing a cup of tea and enjoy [3b1b's video on the topic](https://www.youtube.com/watch?v=3d6DsjIBzJ4).

***

One possible way to use the result and constrain our weights is to approximate $$L(w^{(k+1)})$$ with a first-order Taylor series around $$w^{(k)}$$:

$$
L(w^{(k+1)}) = L(w^{(k)}) + \nabla L(w^{(k)})^\intercal(w^{(k+1)} - w^{(k)})
$$

Substituting this into our loss we get:

$$\begin{align*}

\text{prox}_{L, \lambda}(w^{(k)}) =& \underset{w^{(k+1)}}{\text{argmin}} [ L(w^{(k+1)}) + \lambda \rho(w^{(k+1)}, w^{(k)}) ] \\
\approx& \underset{w^{(k+1)}}{\text{argmin}} [ L(w^{(k)}) + \nabla L(w^{(k)})^\intercal(w^{(k+1)} - w^{(k)}) + \lambda \rho(w^{(k+1)}, w^{(k)}) ] \\
=& \underset{w^{(k+1)}}{\text{argmin}} [\nabla L(w^{(k)})^\intercal w^{(k+1)} + \lambda \rho(w^{(k+1)}, w^{(k)}) ] \\

\end{align*}$$

Note we go from the second to the third line since $$\text{argmin}_{w^{(k+1)}}$$ ignores any terms without $$w^{(k+1)}$$. 

Then similar to the previous section lets computing the gradient with respect to $$w^{(k+1)}$$ and set it to zero:

$$ 
\begin{align*}
0 &= \nabla L(w^{(k)}) + \lambda (w^* - w^{(k)}) \\
w^* &= w^{(k)} - \lambda^{-1} \nabla L(w^{(k)}) \\
\end{align*}
$$

We see that this is gradient descent. Interesting! Notice how if we want our weights to be very close together across iterations then we would set $$\lambda$$ to be a large value so $$\lambda^{-1}$$ would be a small value and so we would be taking very small gradient steps to reduce large changes in our weights across iterations. 

This means that in Euclidean space, proximal policy optimization approximated with a first-order Taylor is the same as regular gradient descent. 

# Use a Second-Order Approximation of $$\rho$$

Another way we could solve the equation is to use the first-order approximation of $$L$$ with a second-order approximation of $$\rho$$. Furthermore, we would let $$\lambda \rightarrow \infty$$ (we want our steps to be as close as possible). 

*Note:* Though we don't need to do this since we solved it directly in the last section, the ideas in this simple example will complement the rest of the post nicely. 

To compute our second order approximation of $$\rho$$ we need to compute $$\nabla \rho$$ and $$\nabla^2 \rho$$.

To do so, we take advantage of the fact $$\lambda \rightarrow \infty$$. This implies that in Euclidean space, $$w^{(k+1)} \approx w^{(k)}$$. And so we get:

$$\rho (w^{(k+1)}, w^{(k)}) = \dfrac{1}{2} \|w^{(k)} - w^{(k+1)} \|^2 \approx \dfrac{1}{2} \|w^{(k)} - w^{(k)} \|^2 = 0$$ 

and 

$$\nabla \rho (w^{(k+1)}, w^{(k)}) = (w^{(k)} - w^{(k+1)}) \approx (w^{(k)} - w^{(k)}) = 0$$

Since both $$\rho$$ and $$\nabla \rho$$ are both $$0$$ when we approx $$\rho$$ with a second-order Taylor series, we are left with only our second order approximation ($$\nabla^2 \rho$$): 

$$\begin{align*}
\rho (w^{(k+1)}, w^{(k)}) &\approx \rho(w^{(k+1)}, w^{(k)}) + \nabla \rho(w^{(k+1)}, w^{(k)})^\intercal (w^{(k+1)} - w^{(k)}) + \dfrac{1}{2} (w^{(k+1)} - w^{(k)})^\intercal G (w^{(k+1)} - w^{(k)}) \\
&= \dfrac{1}{2} (w^{(k+1)} - w^{(k)})^\intercal G (w^{(k+1)} - w^{(k)})
\end{align*}$$

where $$ G = \nabla^2 \rho (w^{(k+1)}, w^{(k)}) \vert_{w^{(k+1)} = w^{(k)}} $$. 

Using the second-order approx of $$\rho$$ with the first-order approx of $$L$$ which we derived last section we get the following: 

$$\begin{align*}
\text{prox}_{L, \lambda}(w^{(k)}) =& \underset{w^{(k+1)}}{\text{argmin}} [ L(w^{(k+1)}) + \lambda \rho(w^{(k)}, w^{(k+1)}) ] \\
\approx& \underset{w^{(k+1)}}{\text{argmin}} [\nabla L(w^{(k)})^\intercal w^{(k+1)} + \lambda \dfrac{1}{2} (w^{(k+1)} - w^{(k)})^\intercal G (w^{(k+1)} - w^{(k)})]
\end{align*}$$

Solving for the optimal $$w^*$$ we get:

$$\begin{align*}
w^* &= w^{(k)} - \lambda^{-1} \text{G}^{-1} \nabla L(w^{(k)}) 
\end{align*}$$

In Euclidean space $$\text{G} = \nabla^2 \rho (w^{(k+1)}, w^{(k)}) = \text{I}$$, so again we get gradient descent:

$$\begin{align*}
w^* &= w^{(k)} - \lambda^{-1} \text{G}^{-1} \nabla L(w^{(k)}) \\
&= w^{(k)} - \lambda^{-1} \text{I}^{-1} \nabla L(w^{(k)}) \\
&= w^{(k)} - \lambda^{-1} \nabla L(w^{(k)}) 
\end{align*}$$

Though this doesn't conclude anything new compared to the previous section, this shows how we can use a second-order approximation for $$\rho$$ to derive an update rule.

# KL Divergence

Now Euclidean space is overall fantastic, but sometimes Euclidean space isn't always that great (example taken from [here](https://wiseodd.github.io/about/)): 

<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/natural_grad/param_space_dist.png" alt="test-acc" class="center"/>
</div>

<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/natural_grad/param_space_dist2.png" alt="test-acc" class="center"/>
</div>

In the two images, we see two Gaussians (coloured orange and blue). 

If we parameterize the Gaussians by their parameters (i.e their mean value $$\mu$$) then both examples have the same distance between them (shown in red). This is called **parameter space**. However, we see that the distributions in the first image are much closer than the ones in the second.

To solve this problem, a better measurement would be to use the KL-Divergence between the distributions. This measures the distance in **distribution space**. Doing so would result in a smaller metric for the first image than the second image, as we would want.

More formally, KL-Divergence (KL-D) measures the distance between two distributions $$p(x \vert \theta')$$ and $$p(x \vert \theta)$$ which are parameterized by $$\theta'$$ and $$\theta$$ and is defined as: 

$$\begin{align*}
D_{KL}[p(x \vert \theta) \| p(x \vert \theta')] &= \sum_x p(x|\theta) [\log \dfrac{p(x \vert \theta)}{p(x \vert \theta')}] \\
&= \sum_x p(x|\theta) [\log p(x \vert \theta) - \log p(x \vert \theta')] \\
&= \mathbb{E}_{p(x \vert \theta)} [\log p(x \vert \theta) - \log p(x \vert \theta')] \\
\end{align*}$$

*Note:* KL-D cannot be applied to neural networks with weights $$w$$ because it measures the distance between *distributions*, which $$w$$ is not. To show this difference we will let $$\theta^{(k)}$$ parameterize some distribution $$p(x \vert \theta^{(k)})$$ at the $$k$$-th step of optimization. 

*Teaser:* We'll see how to apply this to neural nets later ;)

Lets assume we want $$\theta^{(k)}$$ and $$\theta^{(k+1)}$$ to be close across steps. Then we can let $$\rho(\theta^{(k)}, \theta^{(k+1)}) = D_{KL}(p_{\theta^{(k)}} \| p_{\theta^{(k+1)}})$$ where $$p_{\theta^{(k)}} = p(x \vert \theta^{(k)})$$. And so, we will want to solve for:

$$\begin{align*}
\theta^* = \text{prox}_{L, \lambda}(\theta^{(k)}) =& \underset{\theta^{(k+1)}}{\text{argmin}} [ L(\theta^{(k+1)}) + \lambda \rho(\theta^{(k)}, \theta^{(k+1)}) ] \\ 
=& \underset{\theta^{(k+1)}}{\text{argmin}} [ L(\theta^{(k+1)}) + \lambda D_{KL}(p_{\theta^{(k)}} \| p_{\theta^{(k+1)}}) ] \\ 
\end{align*}$$

Similar to last section, lets fisrt approximate $$D_{KL}(p_{\theta^{(k)}} \| p_{\theta^{(k+1)}})$$ with a second-order taylor expansion. Recall from the previous section to approximate $$\rho$$ with a second-order taylor series we need to define $$\rho\vert_{\theta'=\theta}$$, $$\nabla \rho\vert_{\theta'=\theta}$$ and $$\nabla^2 \rho\vert_{\theta'=\theta}$$.  

First, lets derive $$\rho$$, $$\nabla \rho$$, and $$\nabla^2 \rho$$ for $$D_{KL}$$

$$\begin{align*}
\rho = D_{KL}[p(x \vert \theta) \| p(x \vert \theta')] &= \mathbb{E}_{p(x \vert \theta)} [\log p(x \vert \theta) - \log p(x \vert \theta')] \\
&= \mathbb{E}_{p(x \vert \theta)} [\log p(x \vert \theta)] - \mathbb{E}_{p(x \vert \theta)}[\log p(x \vert \theta')] \\
\end{align*}$$

...

$$\begin{align*}

\nabla \rho = \nabla_{\theta'} D_{KL}[p(x \vert \theta) \| p(x \vert \theta')] &= \nabla_{\theta'} \mathbb{E}_{p(x \vert \theta)} [\log p(x \vert \theta)] - \nabla_{\theta'} \mathbb{E}_{p(x \vert \theta)}[\log p(x \vert \theta')] \\
&= - \nabla_{\theta'} \mathbb{E}_{p(x \vert \theta)}[\log p(x \vert \theta')] \\
&= - \nabla_{\theta'} \int p(x \vert \theta) \log p(x \vert \theta') \text{d}x \\
&= - \int p(x \vert \theta) \nabla_{\theta'} \log p(x \vert \theta') \text{d}x

\end{align*}$$

...

$$\begin{align*}
\nabla^2 \rho = \nabla_{\theta'}^2 D_{KL}[p(x \vert \theta) \| p(x \vert \theta')] &= - \int p(x \vert \theta) \nabla_{\theta'}^2 \log p(x \vert \theta') \text{d}x \\
\end{align*}$$

To evaluate $$\rho\vert_{\theta'=\theta}$$, $$\nabla \rho\vert_{\theta'=\theta}$$ and $$\nabla^2 \rho \vert_{\theta'=\theta}$$ we are going to need the two following equations: 

$$\mathbb{E}_{p(x|\theta)} [\nabla_{\theta} \log p(x \vert \theta)] = 0$$

and 

$$\mathbb{E}_{p(x|\theta)} [ \nabla_{\theta}^2 \log p(x \vert \theta) ] = -\text{F}$$

Where $$\text{F} = \mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ \nabla \log p(x \vert \theta) \, \nabla \log p(x \vert \theta)^{\text{T}} \right]$$ is the fisher information matrix. For the full derivation of these eqns checkout [this blog post](https://wiseodd.github.io/techblog/2018/03/11/fisher-information/).

And so using these equations, we get: 

$$\begin{align*}

\nabla_{\theta'} D_{KL}[p(x \vert \theta) \| p(x \vert \theta')] |_{\theta' = \theta} &= - \int p(x \vert \theta) \nabla_{\theta'} \log p(x \vert \theta')|_{\theta' = \theta} \text{d}x \\
&= - \mathbb{E}_{p(x|\theta)} [\nabla_{\theta} \log p(x \vert \theta)] \\
&= 0
\end{align*}$$


$$\begin{align*}
\nabla_{\theta'}^2 D_{KL}[p(x \vert \theta) \| p(x \vert \theta')] |_{\theta' = \theta} &= - \int p(x \vert \theta) \nabla_{\theta'}^2 \log p(x \vert \theta') |_{\theta' = \theta} \text{d}x \\
&= - \mathbb{E}_{p(x|\theta)} [ \nabla_{\theta}^2 \log p(x \vert \theta) ] \\
&= \text{F} \\
\end{align*}$$

Then using the first-order approximation of $$L$$ with our second-order approximation of $$D_{KL}$$ we get:

$$\begin{align*}
\text{prox}_{L, \lambda}(\theta^{(k)}) =& \underset{\theta^{(k+1)}}{\text{argmin}} [ L(\theta^{(k+1)}) + \lambda D_{KL}(p_{\theta^{(k)}} \| p_{\theta^{(k+1)}}) ] \\ 
\approx& \underset{\theta^{(k+1)}}{\text{argmin}} [\nabla L(\theta^{(k)})^\intercal \theta^{(k+1)} + \lambda \dfrac{1}{2} (\theta^{(k+1)} - \theta^{(k)})^\intercal F (\theta^{(k+1)} - \theta^{(k)})] \\
\end{align*}$$

Computing the gradient and setting it to zero we get 

$$\begin{align*}
\theta^* &= \theta^{(k)} - \lambda^{-1}\text{F}^{-1}\nabla L(\theta^k) \\
&= \theta^{(k)} - \lambda^{-1}\tilde{\nabla} L(\theta^k) \\
\end{align*}$$

Where $$\tilde{\nabla} L(\theta^{(k)})$$ is called the **natural gradient**. And this update rule is called **natural gradient descent**. Fantastic! 

Now you're probably thinking:

> "Indeed, this is truly fantastic, we now know how to constrain distributions across iterations using $$D_{KL}$$ but how do we apply it to neural networks with weights $$w$$??"

Patience young padawan, we shall discover this in the following section

# Natural Gradient Descent for Neural Nets 

Though our weights $$w$$ aren't a distribution, usually, our model outputs a distribution $$r( \cdot \vert x)$$. For example, in classification problems like MNIST our model's outputs a probability distribution over the digits 1 - 10. 

The idea is that even though we can't constrain our weights across iterations with KL-D, we can use KL-D to constrain the difference between our output distributions across iterations which is likely to do something similar.

More formally, we can define our output distribution as $$z = f(w, x)$$ and set $$\rho = D_{KL}$$. Then we can define $$\rho_{\text{pull}}$$ as the following:

$$
\rho_{\text{pull}}(w, w') = \mathbb{E}_{x}[f_x * \rho(w, w')] = \mathbb{E}[\rho(f(w, x), f(w', x))]
$$

Where the $$f_x*\rho$$ operator is called a pullback operation: 

$$
f*g(x_1, ..., x_k) = g(f(x_1), f(x_2), ..., f(x_k))
$$

Now similar to the previous sections, lets approximate $$\rho_{\text{pull}}$$ with a second-order Taylor expansion. Again, we need to find $$\rho_{\text{pull}}(w, w')\vert_{w = w'}$$, $$\nabla \rho_{\text{pull}}(w, w')\vert_{w = w'}$$, and $$\nabla_{\text{pull}}^2 \rho(w, w')\vert_{w = w'}$$:

$$\begin{align*}
\rho_{\text{pull}}(w, w')\vert_{w = w'} &= \rho(f(w', x), f(w', x)) \\
&= 0 \\
\end{align*}$$

$$\begin{align*}
\nabla_w \rho_{\text{pull}}(w, w')\vert_{w = w'} &= \nabla_w\rho(f(w, x), f(w', x)) \\
&= \text{J}_{zx}^\intercal \underbrace{\nabla_z\rho(z, z')|_{z=z'}}_{=0} \\ 
&= 0 \\
\end{align*}$$

The last line occurs bc of the chain rule and since we know for $$\nabla \rho = \nabla D_{KL} = 0$$. Notice how pullback functions ($$\rho_{\text{pull}}$$) make the gradients simple to compute :).

Using this decomposition it can be shown that $$\nabla^2 \rho_{\text{pull}}$$ (in general any hessian/second derivative matrix) can be decomposed into the following (check out [this](https://andrew.gibiansky.com/blog/machine-learning/gauss-newton-matrix/) for a full derivation): 

$$\begin{align*}
\nabla_w^2 \rho_{\text{pull}}(w, w') &= \text{J}_{zw}^\intercal H_z \text{J}_{zw} + \underbrace{\sum_a \overbrace{\dfrac{d \rho}{dz_a}}^{=0} \nabla^2_w [f(x, w)]_a}_{=0} \\
&= \text{J}_{zw}^\intercal H_z \text{J}_{zw}
\end{align*}$$

Where $$\text{J}_{zw}$$ is the Jacobian matrix for $$\dfrac{dz}{dw}$$ and $$H_z$$ is the hessian matrix for $$z$$. Again, the second term disappears since we know $$\nabla \rho = \nabla D_{KL} = 0$$.

For the case of $$D_{KL}$$ we know that $$H_z = F_z$$ where $$F_z$$ is the fisher information matrix. And so using the definition of the fisher matrix (i.e $$\text{F} = \mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ \nabla \log p(x \vert \theta) \, \nabla \log p(x \vert \theta)^{\text{T}} \right]$$) we can derive our second-order approximation: 

$$ \begin{align*}
\nabla_w^2 \rho_{\text{pull}}(w, w') = \text{F}_w &= \mathbb{E_{x \sim p_{\text{data}}}} [\text{J}_{zw}^\intercal \text{F}_z \text{J}_{zw}] \\
&= \mathbb{E_{x \sim p_{\text{data}}}} [\text{J}_{zw}^\intercal \mathbb{E}_{t \sim r(\cdot \vert x)} [\nabla_z \log r(t \vert x) \nabla_z \log r(t \vert x) ^ \intercal] \text{J}_{zw}] \\
&= \mathbb{E_{x \sim p_{\text{data}}, {t \sim r(\cdot \vert x)}}} [\nabla_w \log r(t \vert x) \nabla_w \log r(t \vert x) ^ \intercal] \\
\end{align*}$$

Notice $${t \sim r(\cdot \vert x)}$$ samples $$t$$ from the *model's distribution* which is the **true fisher information matrix**. 

Similar, but not the same, is the **empirical fisher information matrix** where $$t$$ is taken to be the true target. 

In total, using our approximation we get the following second-order approximation: 

$$
\rho_{\text{pull}}(w, w') \approx \dfrac{1}{2} (w - w')^\intercal \text{F}_w (w - w')
$$

Which again, if we compute the gradient and set it to zero, we get: 

$$\begin{align*}
w^{(k+1)} &= \underset{w^{(k+1)}}{\text{argmin}} [\nabla L(w^{(k)})^\intercal w^{(k+1)} + \lambda \dfrac{1}{2} (w^{(k)} - w^{(k+1)})^\intercal \text{F}_w (w^{(k)} - w^{(k+1)})] \\

0 &= \nabla L(w^{(k)}) + \lambda (w^{(k)} - w^*)^\intercal \text{F}_w  \\

- \lambda^{-1} \text{F}_w^{-1}\nabla L(w^{(k)}) &=  (w^{(k)} - w^*)   \\
w^* &= w^{(k)} - \lambda^{-1} \text{F}_w^{-1}\nabla L(w^{(k)})   \\

\end{align*}$$

We now see that this results in the **natural gradient method** :o 

# Computing the Natural Gradient in JAX 

The theory is amazing but it doesn't mean much if we can't use it. So, to finish it off we're gonna code it! 

To do so, we'll write the code in **JAX** (what all the cool kids are using nowadays) and train a small MLP model on the MNIST dataset (not the ideal scenario for natural gradient descent to shine but its good to show how everything works).

If you're new to JAX there's a lot of great resources out there to learn from! What do I like about it you ask? Coming from a Computer Science background, I love the functional programming aspect of it which lets you write really clean code :D  

## Code

First, we define a small MLP model 

<script src="https://gist.github.com/gebob19/d0b4e9ef147545b3f5762461c4b2a6fe.js"></script>

Then we define a few loss functions

<script src="https://gist.github.com/gebob19/e288e7a0646d53ee732b6ab9c251b705.js"></script>

Next, we make the losses work for batch input using `vmap`

<script src="https://gist.github.com/gebob19/c2ad9cf658c9fb4a2377614c7dbf8a80.js"></script>

Single example -> batch data with a single line! So clean! :D 

To get a general feel for JAX, we first define a normal gradient step function (we use `@jit` to compile the function which makes it run fast)

<script src="https://gist.github.com/gebob19/d519682cfe29a73f71cdecfc3eb2566b.js"></script>

We can then define a natural gradient step using the **empirical fisher matrix**:

**Recall**: Our update rule is: $$ \theta_{t+1} = \theta_t - \eta F^{-1} \nabla L$$. So, to compute $F^{-1} \nabla L$, we solve the linear system $F x = \nabla L$ using the [conjugate gradient method](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf).

**Aside**: There's a lot of really cool resources to help understand the conjugate gradient method [like this one](https://twitter.com/brennangebotys/status/1410231825868509185?s=20).

<script src="https://gist.github.com/gebob19/2ecb5df93f8f2e7ebb58c0823084bfa6.js"></script>

*Note:* Although we should be using the gradients from `mean_log_likelihood`, we can use the `grads` from the `mean_cross_entropy` to shave off an extra forward and backward pass. *Exercise:* Why does this work? (Hint: does `grad2fisher` change if we use `nll` vs `ll`?)

Finally, we can define the natural gradient step by sampling from the model's predictions: 

*Note:* To evaluate $$\mathbb{E_{t \sim r(\cdot \vert x)}}[...]$$ we use Monte Carlo estimation and sample multiple times from our model distribution which we do using `n_samples`.

<script src="https://gist.github.com/gebob19/ba7d73eb16c8331cc75a7b5dd49e7ab9.js"></script>

We can also checkout the difference in speed (run on my laptop's CPUs) with a batch size of 2: 

<script src="https://gist.github.com/gebob19/3f55c30c8a930b2d4451e1e8baf87d49.js"></script>

`995 µs ± 172 µs per loop (mean ± std. dev. of 7 runs, 1 loop each) # SGD` 

`6.96 ms ± 648 µs per loop (mean ± std. dev. of 7 runs, 1 loop each) # Emp NGD (6x slower SGD)`

`1.33 s ± 21.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) # NGD 1 sample (1000x slower SGD)`

`3.68 s ± 476 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) # NGD 5 sample (3000x slower SGD)`

Last but not least we define some boilerplate training code and compare the runs: 

<script src="https://gist.github.com/gebob19/7200ee0bfc99274b817896cbade5dd8c.js"></script>


Tada! Done! The full code can be found [HERE](https://github.com/gebob19/naturalgradient).

## Results 

**Note:** In the code for the natural gradient (Natural Fisher (1/5 Sample)), the gradients actually blow up (to values up to `1e+20`) to values too big to view on tensorboard (still trainable but results in error lol). I'm not exactly sure why this is yet. But, to get around this, I used gradient clipping (clipped to have values in [-10, 10]). 

***

Here are the tensorboard results: 

### Test Accuracy
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/natural_grad/testacc.png" alt="test-acc" class="center"/>
</div>

### Train Loss
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/natural_grad/loss.png" alt="train-loss" class="center"/>
</div>

### Weights & Gradients

The diagrams in the following order: 

SGD, Empirical Fisher, Natural Fisher (1 Sample), and Natural Fisher (5 Sample).

*Note:* Each diagram uses its own scale. 

#### Linear 0
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/natural_grad/w0.png" class="center"/>
</div>
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/natural_grad/b0.png" class="center"/>
</div>

#### Linear 1
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/natural_grad/w2.png" class="center"/>
</div>
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/natural_grad/b2.png" class="center"/>
</div>

Pretty cool stuff!! :D 

Unfortunately in this scenario, Natural Gradient Descent didn't outperform SGD, but naturally (pun intended) I'm sure there are cases where it does (different learning rates, clipping values, datasets, etc.). 

Did I do anything wrong? Anything to add? What did you like or dislike? 

Let me know -- tweet, email, or dm me! :D 

## Further Reading 

### Problems with Inversion - Conjugate Gradient & Preconditioners

Problems when solving for $F^{-1} \nabla L$ -- Conjugate Gradient Section @  [http://www.depthfirstlearning.com/2018/TRPO](http://www.depthfirstlearning.com/2018/TRPO)

### Problems with Linearlizing

Recall: We linearlize our solution with a Taylor-Approx and then solve for the optimal to make a step. 

Problem: The optimal of the Taylor-Approx may not be the actual optimal. When the optimal is a large step away it may make the update even worse (see diagram below).  

<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/natural_grad/linear_approx_fail.png" alt="test-acc" class="center"/>
</div>

Soln: We can add another term $$\| w^{(k+1)} - w^{(k)}\|^2$$ to make sure we don't take large steps where our approximation is inaccurate. This leads to a dampened update.

### More MOre MORe MORE

[https://www.cs.utoronto.ca/~jmartens/docs/HF_book_chapter.pdf](https://www.cs.utoronto.ca/~jmartens/docs/HF_book_chapter.pdf)

[http://www.cs.toronto.edu/~jmartens/docs/thesis_phd_martens.pdf](http://www.cs.toronto.edu/~jmartens/docs/thesis_phd_martens.pdf)































