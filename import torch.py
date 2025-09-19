import torch
import torch.nn as nn


# fix seed so that random initialization always performs the same 
torch.manual_seed(13)


# create the model N as described in the question
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

# random input
x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # this is required so we can compute the gradient w.r.t x

t = 0 # target class

epsReal = 0.5  #depending on your data this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point erros

# The network N classfies x as belonging to class 2
'''N(x) sends x through the neural network. usually this would reuslt in random values each time because N hasn't been trained yet 
but since we are using a random seed, the value is the same every time. in this case, x is classified as 2 '''

original_class = N(x).argmax(dim=1).item()  # TO LEARN: make sure you understand this expression
print("Original Class: ", original_class)
assert(original_class == 2)
#print(N(x))

# compute gradient
# note that CrossEntropyLoss() combines the cross-entropy loss and an implicit softmax function
L = nn.CrossEntropyLoss()
loss = L(N(x), torch.tensor([t], dtype=torch.long)) # TO LEARN: make sure you understand this line
loss.backward()

# your code here
# adv_x should be computed from x according to the fgsm-style perturbation such that the new class of xBar is the target class t above
# hint: you can compute the gradient of the loss w.r.t to x as x.grad

''' we know what the loss is, we need to find the gradient of the loss,
find the sign of the gradient
multiply with epsilon 
alter input x = x - new value '''
grad = x.grad
n = eps * torch.sign(grad)
adv_x = x - n

new_class = N(adv_x).argmax(dim=1).item()
print("New Class: ", new_class)
assert(new_class == t)
# it is not enough that adv_x is classified as t. We also need to make sure it is 'close' to the original x. 
print(torch.norm((x-adv_x),  p=float('inf')).data)
assert( torch.norm((x-adv_x), p=float('inf')) <= epsReal)

''' for t = 1'''

t = 1
epsReal = 0.2  #depending on your data this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point erros
# get x 
# get N(x)
# compute loss of N(x) and target t = 1
# apply changes to x 
# repeate (get N(x))

# detach to remove excess calcualting needs
# clone to copy x 
# requires grad tells pytroch to keep track of the gradients so that we can use .grad later 
adv_x = x.clone().detach().requires_grad_()

# take 50 steps 
for i in range(50): 
    # calculate loss of input relative to target t 
    loss = L(N(adv_x), torch.tensor([t], dtype=torch.long))
    loss.backward()
    # get gradient of loss function 
    grad = adv_x.grad
    n = eps * torch.sign(grad)
    # take step in loss domain, this results in adv_x being a normal variale type and no longer a leaf
    adv_x = adv_x - n
    # check if new target of chagned x is == t. if it is, then we can stop 
    if N(adv_x).argmax(dim=1).item()  == t and torch.norm(x-adv_x,p=float('inf')) <= 1: 
        break 
    # if not found, need to detach so tat it is not none type
    adv_x = adv_x.detach().requires_grad_()


new_class = N(adv_x).argmax(dim=1).item()
print("new probabilities: ", nn.functional.softmax(N(adv_x),dim=1))

print("New Class (target = 1): ", new_class)
assert(new_class == 1)


print("Norm between x and x': ", torch.norm(x - adv_x,p=float('inf')))
print("x: ", x)
print("x': ", adv_x)

'''IN PART 2, DID AN ITERATIVE APPROACH AND SET EPSILON TO A SMALLER VALUE WHICH RESULTED IN 
X': [ 0.4584,  1.0175, -0.0738,  1.0541, -0.3732,  1.4405, -0.8607,  1.5798,
          0.6597, -0.8023]
L2: 2.4819 

previous value would take steps too large and jump over the threshold where target class is achieved'''
