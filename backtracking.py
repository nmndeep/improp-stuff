'''
    Mathematics for ML Assignment_09 code by Naman Deep Singh, Backtracking line search
    for Gradient descent on a one layer-linear classifier for MNIST data.
'''


import numpy as np
import numpy as np
import scipy.io as sio

                                                    ################      Data loading    ################
mat_contents = sio.loadmat('MNIST-TrainTestNonBinary.mat')
mat_contents.keys()
X_trai = mat_contents['Xtrain']
y_train = mat_contents['Ytrain']
X_test = mat_contents['Xtest']
y_test = mat_contents['Ytest']

X_train = np.hstack((np.array(X_trai),np.array(y_train)))
X_test = np.hstack((np.array(X_test),np.asarray(y_test)))
np.random.shuffle(X_train)

y_train = X_train[:,-1]
y_test = X_test[:,-1]
X_train = X_train[:,:-1]

X = X_train[0:1500, :]
y = np.array(y_train[0:1500],dtype=np.int32)


print("MNIST data using backtracking linesearch")
print("--"*20)

batch_size = 32
n_classes = 10

                    ###################    Gradients and loss calculation   ############

def backprop(X, y, W, lamb):
    F_x = np.dot(X,W)

    C = np.max(F_x, axis=1, keepdims=True)
    exp_scores = np.exp(F_x - C)

    S = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

    #  log-loss
    log_S = -np.log(S[np.arange(batch_size), y])
    L = np.sum(log_S)/batch_size

    # Add regularization using the L2 norm
    L_reg = 0.5*lamb*np.sum(W*W)
    L+= L_reg

    # Gradient of the loss with respect to scores
    grad = S.copy()
    # Substract 1 from the scores of the correct class
    grad[np.arange(batch_size),y] -= 1
    grad /= batch_size

    # Gradient of the loss with respect to weights
    grad_W = X.T.dot(grad)

    # Add gradient regularization
    grad_W+= lamb*W

    return L, grad_W
                                ########   Paramaters and init    #############
alpha = 0.5
beta = 0.8      ###    For backtracking
W = np.zeros(shape=(784, n_classes))
lamb = 1e-2   ### lambda in weight regularizer
t = 1

                                #########     Function for backtracking rule        ############

def backtrack(W, grad_W, loss, alpha, X, y, lamb, t):
    count = 1
    loss_val,_ = backprop(X, y, W-t*grad_W, lamb)
    while loss_val > loss-(t*alpha)*np.sum(grad_W**2):
        t *= beta
    return t
                                #################    Paramaters     ############################
epoch = 5
batch_size = 32
start = 0
lamb = 1e-2

                                ###############     Training with GD and backtracking      #########

while start < 600-batch_size:
    t = 1
    loss, grad_W = backprop(X[start:start+batch_size,:], y[start:start+batch_size], W, lamb)
    alpha = backtrack(W, grad_W, loss, alpha, X[start:start+batch_size,:], y[start:start+batch_size], lamb, t)
    W = W-alpha*grad_W
    start+=batch_size

# val = np.zeros(shape=[550])
corr = 0
predicted = np.argmax(np.dot(X[0:500,:],W))
result = numpy.where(predicted==y[0:predicted.size])
print(result.size)

print("Number of correct, wrong and percent",result.size, 500-result.size,result.size/(500))
