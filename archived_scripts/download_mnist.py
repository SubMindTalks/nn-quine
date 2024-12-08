from torchvision import datasets, transforms

# Set up a directory for the data
data_dir = '../data'

# Download MNIST dataset
mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
