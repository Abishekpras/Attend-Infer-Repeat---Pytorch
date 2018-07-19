from __future__ import print_function
from observations import multi_mnist
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import AIR


def train(epoch, model, train_loader, batch_size, optimizer):
    train_loss = 0
    num_samples = 60000
    for batch_idx, (data, _) in enumerate(train_loader):

        data = Variable(data)

        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss = model(data)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        nn.utils.clip_grad_norm(model.parameters(), clip)

        #printing
        epoch_iters = num_samples // batch_size
        if batch_idx % epoch_iters == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), num_samples,
                100. * batch_idx / epoch_iters,
                kld_loss.data[0] / batch_size,
                nll_loss.data[0] / batch_size))

        train_loss += loss.data[0]


    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / num_samples))


def test(epoch, model, test_loader, batch_size):
    """uses test data to evaluate 
    likelihood of the model"""

    mean_kld_loss, mean_nll_loss = 0, 0
    num_samples = 10000
    for i, (data, _) in enumerate(test_loader):      
        
        data = Variable(data)

        kld_loss, nll_loss = model(data)
        mean_kld_loss += kld_loss.data[0]
        mean_nll_loss += nll_loss.data[0]

    mean_kld_loss /= num_samples
    mean_nll_loss /= num_samples

    print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(mean_kld_loss, mean_nll_loss))

## TODO : Replace with a better generator : Incorporate Shuffling

def train_gen(mnist_train, y_train, batch_size):
    for i in range(mnist_train.shape[0]//batch_size):
        yield mnist_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
        
def test_gen(mnist_test, y_test, batch_size):
    for i in range(mnist_test.shape[0]//batch_size):
        yield mnist_test[i*batch_size:(i+1)*batch_size], y_test[i*batch_size:(i+1)*batch_size]

def fetch_data():
    inpath = 'E:/Docs/Workspace/MS_Thesis_Research/code/data/multi_mnist/'
    (X_train, y_train), (X_test, y_test) = multi_mnist(inpath, max_digits=2, canvas_size=50, seed=42)
    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    X_train /= 255.0
    X_test /= 255.0
    mnist_train = torch.from_numpy(X_train)
    mnist_test = torch.from_numpy(X_test)
    return mnist_train, y_train, mnist_test, y_test

if __name__== "__main__":
    
    #hyperparameters
    n_epochs = 100
    clip = 10
    learning_rate = 1e-3
    batch_size = 64
    seed = 128

    #manual seed
    torch.manual_seed(seed)
    plt.ion()

    model = AIR()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mnist_train, y_train, mnist_test, y_test = fetch_data()
    train_loader = iter(train_gen(mnist_train, y_train, batch_size))
    test_loader = iter(test_gen(mnist_test, y_test, batch_size))

    for epoch in range(1, n_epochs + 1):

        #training + testing
        train(epoch, model, train_loader, batch_size, optimizer)
        test(epoch, model, test_loader, batch_size)

        #saving model
        if epoch % 10 == 1:
            fn = 'data/air_state_dict_'+str(epoch)+'.pth'
            torch.save(model.state_dict(), fn)
            print('Saved model to '+fn)
