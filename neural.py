from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as anim
import tensorflow as tf
import numpy as np
import sys
from copy import deepcopy
import time


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def check(self):
        return time.time() - self.start

    def __exit__(self, *args):
        end = time.time()
        self.interval = end - self.start
        
def rot180(x):
    return np.rot90(np.rot90(x))

def is_square(x):
    try:
        return x.shape[0] == x.shape[1]
    except Exception:
        return False

def make_vec(x):
    return x.reshape((-1, ))

def xavier(dims):
    return np.random.randn(dims[0], dims[1])/np.sqrt(dims[0]/2)

def normalise(x):
    try:
        if len(x.shape) == 1 or x.shape[1] == 1:
            x -= np.mean(x)
            stddev = np.var(x)
            x /= stddev
    except Exception:
        res = x
        for i in range(x.shape[1]):
            res[i, :] -= np.mean(x[i, :])
            res[i, :] /= np.var(res[i, :])
        return res
    res = x.astype('float64')
    res.reshape(-1, 1).astype('float64')
    res -= np.mean(res).astype('float64')
    res /= np.mean(x)
    return res.reshape(x.shape)

def ReLU(x):
    res = x
    res[np.where(res < 0)] *= 0.01
    return res

def ReLU_prime(x):
    res = np.ones_like(x)
    res[np.where(x < 0)] = 0.01
    return res

def make_square(x):
    if not is_square(x):
        try:
            return x.reshape(int(np.sqrt(x.shape[0]*x.shape[1])), -1)
        except Exception:
            return x.reshape(int(np.sqrt(x.shape[0])), -1)

def identity(x):
    return x

def identity_prime(x):
    return np.ones_like(x)

def cross_entropy_loss(label_vec, predicted_vec):
    loss = 0
    for i in range(len(label_vec)):
        loss -= label_vec[i]*np.log2(predicted_vec[i])
    return loss

def sq_err(label_vec, predicted_vec):
    return np.sum(np.multiply(label_vec-predicted_vec, label_vec-predicted_vec))

def softmax(activations):
    ex = np.exp(activations)
    denominator = np.sum(ex)
    return ex/denominator

class fully_connected_layer:
    
    def __init__(self, dims, a_fn, a_prime_fn, final=False):
        self.w = xavier(dims)
        self.b = np.random.randn(self.w.shape[0])/np.sqrt(dims[1]/2)
        self.z = np.zeros((self.w.shape[0], ))
        self.a = np.zeros((self.w.shape[0], ))
        self.sigma = a_fn
        self.sigma_prime = a_prime_fn
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)
        self.inputs = np.zeros((self.w.shape[1], ))
        self.final = final
        self.normalfn = identity if self.final else normalise
        self.delta = np.zeros_like(self.a)
        self.vb = np.zeros_like(self.b)
        self.vw = np.zeros_like(self.w)
        self.name = 'Fully connected layer'
        
        
    def forward(self, inputs):
        self.z = self.w @ make_vec(inputs) + self.b
        self.a = self.sigma(self.z)
        
    def backward(self, delta=None, output=None, labels=None, w=None):
        if self.final:
            self.delta = output - labels
        else:
            self.delta = np.multiply(w.T @ delta,
                                     self.sigma_prime(self.z))
    
    def update_grads(self, inputs):
        self.grad_w += np.outer(self.delta, inputs.reshape(-1,))
        self.grad_b += self.delta
        
    def apply_grads(self, lr, bs, rho=0):
        self.vw = rho * self.vw + self.grad_w*lr/bs
        self.vb = rho * self.vb + self.grad_b*lr/bs
        self.w -= self.vw
        self.b -= self.vb
        self.grad_w = np.zeros_like(self.grad_w)
        self.grad_b = np.zeros_like(self.grad_b)
    
# THIS IS UNFINISHED
class convolutional_layer:
    """
    UNFINISHED
    """
    def __init__(self, input_size, stride, kernel_size, a_fn, a_prime_fn,
                 final=False, padding='SAME'):
        self.p = (kernel_size-1)/2
        self.k = kernel_size
        self.s = stride
        self.input_size = input_size
        self.output_size = int((self.input_size + 2*self.p - self.k)/self.s) + 1
        
        self.w = np.random.rand(self.k, self.k).astype('float64')
        self.b = np.ones((1, ))*np.random.rand(1)[0]
        self.z = np.zeros((self.output_size, self.output_size))
        self.a = np.zeros_like(self.z)
        self.sigma = a_fn
        self.sigma_prime = a_prime_fn
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = 0
        self.inputs = np.zeros((input_size, input_size))
        self.final = final
        self.normalfn = identity if self.final else normalise
        self.delta = np.zeros_like(self.a)
        self.name = 'Convolutional layer'
        
    def forward(self, inputs):
        inputs = make_square(inputs)
        self.z = convolve(inputs, self.w, self.s, self.p, self.b)
        self.a = self.sigma(self.z)
                
    def backward(self, delta, w):
        print(delta.shape)
        print(w.shape)
        delta = make_square(delta)
        w = make_square(w)
        p = 0.5*(self.output_size + self.w.shape[0] - delta.shape[0] - 1)
        print(delta.shape)
        print(w.shape)
        self.delta = np.multiply(convolve(delta, rot180(w), 1, p, 0),
                                 self.sigma_prime(self.z))
        
    def update_grads(self, inputs):
        if inputs.shape[0] != inputs.shape[1]:
            inputs.reshape((int(np.sqrt(inputs.shape[0]))), -1)
        self.grad_w += np.outer(self.delta, inputs)
        self.grad_b += self.delta
        
    def apply_grads(self, lr, bs, rho=0):
        self.vw = rho * self.vw + self.grad_w*lr/bs
        self.vb = rho * self.vb + self.grad_b*lr/bs
        self.w -= self.vw
        self.b -= self.vb
        self.grad_w = np.zeros_like(self.grad_w)
        self.grad_b = np.zeros_like(self.grad_b)
        

class neural_net:
    
    def __init__(self, train_x, train_y, test_x, test_y, 
                 init_type='random', losstype='acc'):
        
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.processed_data = 0
        self.max_output = []
        self.losstype = losstype
        
        self.layers = []
        self.train_acc = []
        self.processed_arr = []
        self.test_acc = []
        
    def add_layer(self, layer):
        if len(self.layers) > 0:
            self.layers[-1].final = False
            self.layers[-1].normalfn = normalise
        self.layers.append(layer)
        self.layers[-1].final = True
        self.layers[-1].normalfn = identity
    
    def forward_prop(self, x):       
        self.layers[0].forward(x)
        for idx in range(1, len(self.layers)):
            self.layers[idx].forward(self.layers[idx-1].a)
        self.output_layer = self.layers[-1].a
    
    def back_prop(self, x, labels):
        self.layers[-1].backward(delta=None, output=self.output_layer,
                                 labels=labels)
        for idx in range(len(self.layers) - 2, -1, -1):
            self.layers[idx].backward(delta=self.layers[idx+1].delta,
                                      w=self.layers[idx+1].w)
            
        
            
    def update_grads(self, x):
        self.layers[0].update_grads(x)
        for idx in range(1, len(self.layers)):
            self.layers[idx].update_grads(self.layers[idx-1].a)
            
    def apply_grads(self, lr, bs):
        for layer in self.layers:
            layer.apply_grads(lr, bs, rho=0.9)
                        
    def train(self, epochs, batch_size, lr):
        for epoch in range(epochs):
            samples = np.random.choice(self.train_x.shape[0],
                                       self.train_x.shape[0],
                                       replace=False)
            n_iters = int(train_x.shape[0]/batch_size)
            train_acc = ""
            test_acc = ""
            for b_idx in range(n_iters):
                batch = samples[b_idx*batch_size:(b_idx+1)*batch_size]
                for i in range(len(batch)):
                    percent = 100*self.processed_data/self.train_x.shape[0]/epochs
                    trained_str = str(self.processed_data)
                    percent_str = str(percent).split('.')[0] + '%'
                    sys.stdout.write(
                        "\rTraining: " + percent_str +
                        ", training samples processed: " + trained_str +
                        ", training accuracy: " + train_acc + 
                        ", testing accuracy: " + test_acc)
                    sys.stdout.flush()
                    x = self.train_x[batch[i], :]
                    y = self.train_y[batch[i], :]
                    self.forward_prop(x)
                    self.max_output.append(np.amax(self.output_layer))
                    self.back_prop(x, y)
                    self.update_grads(x)
                    if self.processed_data % 2000 == 0:
                        test_acc = self.test(1000, 'test')
                        train_acc = self.test(1000, 'train')
                        self.processed_arr.append(self.processed_data)
                        self.train_acc.append(train_acc)
                        self.test_acc.append(test_acc)
                        if self.losstype == 'acc':
                            ans = str(test_acc).split('.')[0]
                            pad = ' '*(3 - len(ans))
                            test_acc = pad + str(test_acc).split('.')[0] + "%"
                            ans = str(train_acc).split('.')[0]
                            pad = ' '*(3 - len(ans))
                            train_acc = pad + str(train_acc).split('.')[0] + "%"
                        else:
                            ans = str(test_acc)[:5]
                            pad = ' '*(3 - len(ans))
                            test_acc = pad + ans
                            ans = str(train_acc)[:5]
                            pad = ' '*(3 - len(ans))
                            train_acc = pad + ans
                    self.processed_data += 1
                    
                self.apply_grads(lr, batch_size)
                
        sys.stdout.write(
            "\rTraining: 100%, test accuracy: " + test_acc + "\n")
            
    def test(self, n, mode):
        wrong = 0
        samples = np.random.choice(self.test_x.shape[0], n, replace=False)
        loss = 0
        for i in range(n):
            x = None
            if mode == 'test':
                x = self.test_x[samples[i], :]
                y = self.test_y[samples[i], :]
            else:
                x = self.train_x[samples[i], :]
                y = self.train_y[samples[i], :]
            self.forward_prop(x)
            if self.losstype == 'acc':
                pred = np.argmax(self.output_layer)
                if y[pred] == 0:
                    wrong += 1
            else:
                loss += sq_err(y, self.output_layer)
                
        if self.losstype == 'acc':
            return 100 * (n - wrong)/n
        else:
            return loss/n
    
    def classify(self, imgpath=None, img=None):
        try:
            img.shape
        except Exception:
            img = normalise(mpimg.imread(imgpath)[:, :, 0])
        img.reshape(-1, 1)
        self.forward_prop(img)
        return np.argmax(self.output_layer)
                
    @property
    def info(self):
        print("\nActivations:")
        for layer in self.layers:
            print(layer.name, layer.a.shape)
        print("------------")
        print("Weights:")
        for layer in self.layers:
            print(layer.name, layer.w.shape)
        print("------------")
        print("Biases:")
        for layer in self.layers:
            print(layer.name, layer.b.shape)
        print("------------")
        print("Deltas:")
        for layer in self.layers:
            print(layer.name, layer.delta.shape)
            
    
    @property
    def maxmins(self):
        print("\nActivations:")
        for layer in self.layers:
            print(np.amin(layer.a), np.amax(layer.a))
        print("------------")
        print("Weights:")
        for layer in self.layers:
            print(np.amin(layer.w), np.amax(layer.w))
        print("------------")
        print("Biases:")
        for layer in self.layers:
            print(np.amin(layer.b), np.amax(layer.b))
        print("------------")
        print("Deltas:")
        for layer in self.layers:
            print(np.amin(layer.delta), np.amax(layer.delta))
            
    def plot(self):
        plt.plot(self.processed_arr, self.train_acc, 'r-')
        plt.plot(self.processed_arr, self.test_acc, 'b-')
        if self.losstype == 'acc':
            plt.ylim([0, 100])
        else:
            plt.ylim([0, 1.5])
        ylab = 'Accuracy (%)' if self.losstype=='acc' else 'Cross entropy loss ' + r'(-$\Sigma_i p_i \ln{q_i}$)'
        plt.ylabel(ylab)
        plt.xlabel('Data processed')
        title = 'Test and training accuracy on MNIST' if self.losstype == 'acc' else 'Test and training loss on MNIST'
        plt.title(title)
        plt.legend({'Training', 'Testing'})
            
    def plot_update(self):
        self.h1.set_xdata(self.processed_arr)
        self.h2.set_xdata(self.processed_arr)
        self.h1.set_ydata(self.train_acc)
        self.h2.set_ydata(self.test_acc)
        plt.show()
        try:
            plt.gcf().canvas.flush_events()
        except NotImplementedError:
            pass

def get_data():
  return input_data.read_data_sets("MNIST_data/", one_hot=True)

mnist = get_data()
mnist_eval = get_data()
train_x = mnist.train.images
test_x = mnist.test.images
train_y = mnist.train.labels
test_y = mnist.test.labels

for data in [train_x, test_x]:
    mean_img = np.mean(data, axis=0)
    for i in range(data.shape[0]):
        data[i, :] -= mean_img

nn = neural_net(train_x, train_y, test_x, test_y, init_type='random',
                losstype='loss')

nn.add_layer(fully_connected_layer((64, 28*28), ReLU, ReLU_prime))
nn.add_layer(fully_connected_layer((64, 64), ReLU, ReLU_prime))
nn.add_layer(fully_connected_layer((32, 64), ReLU, ReLU_prime))
nn.add_layer(fully_connected_layer((16, 32), ReLU, ReLU_prime))
nn.add_layer(fully_connected_layer((10, 16), softmax, None))
   
nn.train(epochs=10, batch_size=32, lr=1e-3)
nn.plot()

def classify(nn, imgpath=None, img=None):
    try:
        img.shape
    except Exception:
        img = normalise(mpimg.imread(imgpath)[:, :, 0]).astype('float64')
    img.reshape(-1, 1)
    nn.forward_prop(img)
    return np.argmax(nn.output_layer)
