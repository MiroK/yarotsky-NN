# Yarotsky's Neural Networks

Implementation of ReLU neural networks from the paper

```
Error bounds for approximations with deep ReLU networks  arXiv:1610.01145
```

The idea is to implement the function $f_m$ approximating $x^2$ in different ways.
First check with everyting by implementing the non-NN solution. Then
we try different architectures and fix the weights our selves. This way
we can obtain e.g. a fully connected NN, NN with skip connections or futher
optimize and share weights.

Each of these networks should finally be implemented in tensor flow and the
weights learned by training. The ultimate question is whether the Yaratosky
net can be beaten / how close to it can we get by training (or manually).

# TODO
- [ ] in `tooth.py` implement `x2_approx_noskip`, i.e. as fully connected NN
- [ ] train the tensorflow networks
- [ ] parallel training
