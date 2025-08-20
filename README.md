# Orth_grad

This is a optimizer inspired directly from Muon but with completely different methods. It reached 92% accuracy on CIFAR-10 with the network in Network-Example within 30 epochs.

I haven't do many experiment yet because I don't have any stronger GPUs but my Nvidia Geforce 4060 on my laptop, which will take me over minutes to run a 30 epochs experiment on CIFAR-10. I will be very happy if anyone could do some extra experiment on it on your spare GPUs.

The optimizer use both Momentum from Adam and also a orthogonal gradient (rather than momentum) with curvature to accerlerate optimization.
