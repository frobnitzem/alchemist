
After the initial training code was written, I added some extra output
on the geometry of the flow learned during training.  I opted to output the force evaluated at the embdding points.  However, mapping the whole flow field should be possible here.  Alternately, we could show input -> output pairs for points like the initial embeddings.

Testing the training loop on a small problem (3 atom types embedded
to points of an equilateral triangle), I found no significant decrease in the loss function during training (train1.csv).  After seeing the same behavior with a different optimizer (switched SGE to Adam), and when raising the learning rate, I concluded that there must be something wrong with the data.

Looking at the generated embeddings showed they overlapped one another substantially.  So, I decreased the variance of the MixtureNormal step.  This yielded loss functions that decreased.  Increasing the number of batches again gave train2.csv.  This still doesn't show a convergence behavior.

So, I went back to the basic idea of mapping distribution rho\_0
onto distribution rho\_1.  The mapping needs to consider the
spatial support of both distributions.  For example, we may have
rho\_1(x) = rho\_0(10-x) -- flipped around x=5.  This could be done in one step given an arbitrary affine transform.  However, it is not possible within a few integration timesteps of a leapfrog integrator.  The underlying problem is that we don't have a discrete mapping that would allow jumps, but a continuous change.  It's probably a good idea to consider Monte-Carlo like jump moves within the mapping.  For now, I increased the number of timesteps to 100, with a dt value of 0.01.  The resulting train3.csv finally shows some minimization behavior.  However, as we've seen, the particle type mapping problem is nontrivial on its own.

Now that it's working, looking at the force output for the embedding
points shows a convergence behavior there as well.  It would be good
to plot the entire energy landscape, as well as the initial -> final
point mapping of each of the Gaussian distributions from their embedding
point.  Finally, we should do the same plot for the reverse time
evolution of the points on H0.  I suspect the time/momentum symmetry
makes the potential energy landscape flat, or sloped toward a high-probability point.  This is not necessarily optimal, since we want the system
to migrate toward the origin in the 1->0 direction and take the opposite motion in the 0->1 direction.  However, this ends up being momentum-dependent in the leapfrog scheme.

We should investigate a purely diffusive scheme, where the loss is
some clever evaluation over trajectories in coordinate space
(no momenta).  Then reversing the time direction would be more
impactful.  Alternately, we should make time a formal
parameter of the Hamiltonian and show this leads to directional
migration.

