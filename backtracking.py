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

X = X_train[:, :]
y = np.array(y_train,dtype=np.int32)


print("MNIST data using backtracking linesearch")
print("--"*20)

batch_size = 100
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
alpha = 0.4
beta = 0.6      ###    For backtracking
W = np.zeros(shape=(784, n_classes))
lamb = 0   ### lambda in weight regularizer
t = 1

                                #########     Function for backtracking rule        ############

def backtrack(W, grad_W, loss, alpha, X, y, lamb, t):
    count = 1
    loss_val,_ = backprop(X, y, W-t*grad_W, lamb)    #####     Function value at x-t*x'
    while loss_val <= loss-(t*alpha)*np.square(np.linalg.norm(grad_W, 'fro')):              ########      f(x-tx') > f(x) - t*alpha*|x'|^2
        t *= beta
    return t
                                #################    Paramaters     ############################
batch_size = 100

                                ###############     Training with GD and backtracking      #########
start=0
alpha = 1.0
count = 1
grad_W = np.zeros_like(W)
t = 1
                              #################      Condition either full run of training data or ||grad_W||<1e-3      ##########
while np.linalg.norm(grad_W, 'fro') < 1e-3:
    start = 0
    while start < 60000-batch_size :
        loss, grad_W = backprop(X[start:start+batch_size,:], y[start:start+batch_size], W, lamb)
        alpha = backtrack(W, grad_W, loss, alpha, X[start:start+batch_size,:], y[start:start+batch_size], lamb, t)
        # count+=1
        t = alpha
        W = W - alpha*grad_W
        start+=batch_size

######        Training  and Test Accuracy      ###########
predicted = np.argmax(np.dot(X,W), axis = 1)
result = np.where(predicted==y[0:predicted.size])
print("Number of correct, wrong and percentage for trainig data",result[0].size, predicted.size-result[0].size,result[0].size/predicted.size)
predicted = np.argmax(np.dot(X_test[:,:-1],W), axis = 1)
result = np.where(predicted==y_test[0:predicted.size])
print("Number of correct, wrong and percentage for test data",result[0].size, predicted.size-result[0].size,result[0].size/predicted.size)
