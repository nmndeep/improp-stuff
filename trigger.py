'''
	FGSM is tried in this code. for l-inf 10 error is at ~70%.  
	This is done by creating data by all combinations of given ranges tot_points = 35*12*27*73
'''
import torch
import r4l.agent as agent
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

nw = agent.Network([16,16])
sd = torch.load('../pth/barto_small_20000_2_16_deterministic_randomstart.pth')
nw.load_state_dict(sd)
test_input = torch.tensor((1,2,1,1), dtype=torch.float)


batch_size = 5000
classes = 9
x_ = np.arange(0,35)
y_ = np.arange(0,12)
dy_ = np.arange(-13,14)
dx_ = np.arange(-36,37)

def populate():

###    Create all combinations in the given resp. ranges and calculate original model predictions

	fin_tens1 = np.array(np.meshgrid(x_,y_,dx_,dy_)).T.reshape(-1,4)
	tensor_x = torch.from_numpy(fin_tens1)
	tensor_x = tensor_x.float()
	batch_out = torch.empty(tensor_x.size()[0])
	raw_logits = torch.empty((tensor_x.size()[0], classes))

	permutation = torch.randperm(tensor_x.size()[0])

####    Get original predictions

	for i in range(0,tensor_x.size()[0], batch_size):

		indices = permutation[i:i+batch_size]
		pred = nw(tensor_x[indices])
		act = torch.argmax(pred, dim=1)
		batch_out[indices] = act.float()
		raw_logits[indices] = pred

	torch.save(batch_out, './y_labels.pt')
	
	return tensor_x, batch_out, raw_logits



def fgsm(model, X, y, epsilon, retain = True):

    """ Construct FGSM adversarial examples on the examples X"""

    delta = torch.zeros_like(X, requires_grad=True)
    loss = torch.nn.CrossEntropyLoss()(model(X+ delta), y)  ##   the agent net uses mse_loss
    loss.backward()
    return epsilon * (delta.grad.detach().sign())    ###   fgsm is epsilon * (delta.grad.detach().sign())

if __name__ == '__main__':

	l = torch.tensor([[x_[0], y_[0], dx_[0], dy_[0]]])   ###   x, y, dx, dy  Ranges for clamping the values
	u = torch.tensor([[x_[-1], y_[-1], dx_[-1], dy_[-1]]])

	correct = 0
	tot = 0
	max_norm = 0
	tensor_x, tensor_y, y_logits = populate()
	permutation = torch.randperm(tensor_x.size()[0])
	# targeted = torch.ones_like(tensor_y)*7
	for i in range(0, tensor_x.size()[0], batch_size):

		indices = permutation[i:i+batch_size]

		# one_hot = torch.nn.functional.one_hot(tensor_y[indices].to(torch.int64),9)  
		delta = fgsm(nw, tensor_x[indices], tensor_y[indices].type(torch.LongTensor), 10)  

		clipped_x = torch.max(torch.min(tensor_x[indices] + delta, u), l)
		y_pert = torch.argmax(nw(clipped_x), dim = 1)

		vals = torch.abs(clipped_x-tensor_x[indices])
		max_norm = max_norm if torch.max(vals).item() <=max_norm else torch.max(vals).item()
		correct += (y_pert == tensor_y[indices]).sum().item()
		tot+=vals.size()[0]

	print("Error: {:.2f}".format((1-correct/tot) * 100))
	print("Max linf_norm: {:.2f}".format(max_norm))
