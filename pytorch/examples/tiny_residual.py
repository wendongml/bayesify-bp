"""One model supports mean prediction, moment propagation, and graph inspection."""

import torch

from bayesify_torch import DiagonalMoments, TinyResidualMLP, mfvi_categorical_head
from bayesify_torch.inspect import fx_graph, keras_like_summary


torch.manual_seed(7)
model = TinyResidualMLP()
x = torch.randn(5, 4)
belief = DiagonalMoments(x, torch.zeros_like(x))
forward = model.forward_moments(belief)
head = mfvi_categorical_head(forward.output, torch.tensor([0, 1, 2, 1, 0]))
messages = forward.backward(head.signal)

print(model)
print(keras_like_summary(model, x))
print(fx_graph(model).graph)
print("readout shape:", tuple(forward.output.mean.shape))
print("parameter messages:", sorted(messages.parameters))
