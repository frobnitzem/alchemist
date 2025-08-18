# Embedding Distributions for Labels

In generative AI, the goal is to create some instance of
a set with prescribed characteristics.  Often, we assign
labels to distinguish sub-categories within our dataset.

Normalizing flow models can be setup to evolve category
labels together with the flow.  For example, we could
generate games on a shelf, organized by highest age
games at the top, and lowest age games at the bottom.
Or, we could generate buildings on a city map, with banks
near insurance and real estate companies, and
apartments near restaurants.

In this note, we apply Bayes' theorem to
derive a training loss function for joint distributions
of categories and positions like the above.
We give several useful properties about the
distribution.  We then present results on training
generative models for the use cases above.


Argmax Flows were described by Hoogeboom 2021 and in
[Hoogeboom's tutorial](https://ehoogeboom.github.io/post/en_flows/)
are given as

$$
P(z | h) = \begin{cases}
    1, & z = argmax(h) \\
    0, & \text{o.w.}
\end{cases}
$$

Based on this, Hoogeboom describes the sampling distribution
for observing $z$ with an ELBO loss,
$$
\log P(z) = \log \mathcal E_{h \sim q(h|z)} \left[
   P(h)P(z|h)/q(h|z)
\right]
\ge \mathcal E_{h \sim q(h | z)} \left[
   \log P(h) - \log q(h|z)
\right]
$$

With the trial distribution, $q(h' | z)$ as normally distributed,
followed by a change of variable to
$h = h' - (1-e_z) \mathrm{softplus}(h'-h_z')$.
Here, $e_z$ is a one-hot encoding of $z$
-- a unit vector in the direction $z$.
Softplus of $x$ is $\log(1+e^x)$.
This final change of variable is suitable for cases where $z$ is
the index of the largest value of $h$.

However, a geometric structure can be used instead, where
the category is determined by the closest embedding vector.
The spirit of generative neural networks is that we do
not impose a structure on $P(h)$.  However, if
the distribution of $h$ is a Gaussian centered around
$h_z$, then we would have

$$
q(h|z) = exp(-(h-h_z)^2/2\sigma^2) (2\pi\sigma^2)^{-\mathrm{dim}(h)/2}
$$

The key idea is that the underlying probability distribution over $z$ is
divided into zones,

$$
P(z|h) = \frac{q(h|z) q(z)}{\sum_k q(h|k) q(k)} \\
   = \frac{1}{1 + \sum_{k\ne z}
exp((h-[h_k+h_z]/2)\cdot(h_k-h_z)/sigma^2) q(k)/q(z)}
$$

This structure is much more intuitive than simply taking the argmax,
since nearby embeddings compete for label space.


In the generative model, we need to train some
transformation of $h$ from an initial distribution
to P(h), which is unknown.  Because P(h) is unknown,
P(z) is also unknown, and so we should assign
$$
\log P(z) = \log \mathcal E_{h \sim q(h|z)} \left[
   P(h)P(z|h)/q(h|z)
\right]
\ge \mathcal E_{h \sim q(h | z)} \left[
   \log P(h) + \log P(z|h) - \log q(h|z)
\right]
$$

The training process from a given $z$ then goes as follows.
First, draw $h\sim q(h|z)$. Then run the backward process
from $h$ to $h_0$ to compute $P(h)$.  Then, the loss for this
sample is $-\log P(h) - \log P(z|h) + \log q(h|z)$.

