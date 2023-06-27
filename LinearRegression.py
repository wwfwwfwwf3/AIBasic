import torch
import matplotlib.pyplot as plt
import numpy as np

X=torch.arange(-5,5,0.1).view(-1,1)
func=-X*5

Y=func+0.4*torch.randn(X.size())
b = torch.tensor(-20.0, requires_grad = True)

# Plot and visualizing the data points in blue
plt.plot(X.numpy(), Y.numpy(), 'b+', label='Y')
plt.plot(X.numpy(), func.numpy(), 'r', label='func')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

def forward(x):
    return w*x+b

# evaluating data points with Mean Square Error.
def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)

w = torch.tensor(-10.0, requires_grad=True)

step_size = 0.1
loss_list = []
iter = 20

for i in range (iter):
    # making predictions with forward pass
    Y_pred = forward(X)
    # calculating the loss between original and predicted data points
    loss = criterion(Y_pred, Y)
    # storing the calculated loss in a list
    loss_list.append(loss.item())
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    # updateing the parameters after each iteration
    w.data = w.data - step_size * w.grad.data
    b.data = b.data - step_size * b.grad.data
    # zeroing gradients after each iteration
    w.grad.data.zero_()
    b.grad.data.zero_()
    # priting the values for understanding
    print(f'Iteration: {i}, loss: {loss.item()}, w: {w.item()}, b: {b.item()}')

# Plotting the loss after each iteration
plt.plot(loss_list, 'r')
plt.xlabel("Epochs/Iterations")
plt.ylabel("Loss")
plt.show()

# Plotting the predicted and original data points
Y_hat = forward(X)
plt.plot(X.numpy(),Y_hat.detach().numpy(),label='Predicted')
plt.plot(X.numpy(),Y.numpy(),'r+',label='Original')
plt.show()




