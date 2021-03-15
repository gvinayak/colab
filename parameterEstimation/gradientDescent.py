import numpy
import theano
import theano.tensor as T
from theano import pp

x = T.dscalar('x')
y = x ** 2
gy = T.grad(y, x)
print pp(gy)  # print out the gradient prior to optimization
#'((fill((x ** TensorConstant{2}), TensorConstant{1.0}) * TensorConstant{2}) * (x ** (TensorConstant{2} - TensorConstant{1})))'
f = theano.function([x], gy)
print f(4)
print numpy.allclose(f(94.2), 188.4)
print pp(f.maker.fgraph.outputs[0])


