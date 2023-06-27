import torchvision.datasets as datasets

minist_trainset=datasets.MNIST(root='.\data', train=True, download=True)
print(minist_trainset)