import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from torch import optim
from sklearn.metrics import accuracy_score

EPS = 1e-10

class Leaf(nn.Module):
    def __init__(self, i_size, o_size, h_size=128):
        super(Leaf, self).__init__()
        self.i2h = nn.Linear(i_size, h_size)
        self.h2o = nn.Linear(h_size, o_size)
        self.soft = nn.LogSoftmax(1)
        self.relu = nn.ReLU()
        self.is_leaf = True

    def forward(self, features):
        out = self.i2h(features)
        out = self.relu(out)
        out = self.h2o(out)
        return self.soft(out)

    def accum_probs(self, features, path_prob):
        return [[path_prob, self.forward(features)]]

    def calc_regularization(self, features, path_prob):
        return 0

class Node(nn.Module):
    def __init__(self, i_size, o_size):
        super(Node, self).__init__()
        self.o_size = o_size
        self.i_size = i_size
        self.i2o = nn.Linear(i_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.is_leaf = False
    
    def build_tree(self, depth):
        if depth - 1 < 0:
            raise ValueError("Depth must be greater than zero.")
        if depth - 1 > 0:
            self.left = Node(self.i_size, self.o_size)
            self.right = Node(self.i_size, self.o_size)
            self.left.build_tree(depth - 1)
            self.right.build_tree(depth - 1)
        else:
            self.left = Leaf(self.i_size, self.o_size)
            self.right = Leaf(self.i_size, self.o_size)

    def forward(self, features):
        pr = self.prob_left(features)
        return pr*self.left(features) + (1 - pr)*self.right(features)

    def prob_left(self, features):
        return self.sigmoid(self.i2o(features))

    def accum_probs(self, features, path_prob):
        res = []
        p_l = self.sigmoid(self.i2o(features)).squeeze()
        res_l = self.left.accum_probs(features, p_l*path_prob)
        res_r = self.right.accum_probs(features, (1 - p_l)*path_prob)
        res.extend(res_l)
        res.extend(res_r)
        return res

    def calc_regularization(self, features, path_prob):
        p_l = self.prob_left(features).squeeze()
        alpha = (path_prob*p_l).sum()/(path_prob.sum())
        C_here = -0.5*torch.log(alpha + EPS) - 0.5*torch.log(1 - alpha + EPS)
        C = self.left.calc_regularization(features, p_l*path_prob) + \
                        self.right.calc_regularization(features, (1 - p_l)*path_prob)
        C = C + C_here
        return C
        

def tree_loss(path_probs, y_true, C, gamma):
    loss = 0
    criterion = nn.NLLLoss(reduce=False)
    for p, pred in path_probs:
        loss += (p*criterion(pred, y_true)).mean()
    return loss.mean() + gamma*C

def tree_logloss(path_probs, y_true):
    """
    Original loss from paper
    """
    loss = 0
    criterion = nn.NLLLoss()
    for p, pred in path_probs:
        loss -= (p.squeeze()*criterion(pred, y_true)).mean()
    return -torch.log(loss.mean())

def train(model, batches_train, batches_val, n_epoch=5, gamma=0.1,
          criterion=tree_loss, val_every=500, print_every=100):
    model.train()
    optimizer = optim.Adam(model.parameters())
    all_losses = np.zeros(print_every)
    plot_train = []
    plot_val = []
    for epoch in range(n_epoch):
        print('Epoch: {}'.format(epoch))
        for i, batch in enumerate(batches_train):
            optimizer.zero_grad()
            features, targets = batch
            features = Variable(features.view(-1, 28*28))
            targets = Variable(targets)
            prb = model.accum_probs(features, Variable(torch.Tensor([1]*targets.shape[0])))
            C = model.calc_regularization(features, Variable(torch.Tensor([1]*targets.shape[0])))
            loss = tree_loss(prb, targets, C, gamma=gamma)
            loss.backward()
            optimizer.step()
            plot_train += [loss.data[0]]
            all_losses[(i + 1)%print_every] = loss.data[0]
            if (i + 1) % print_every == 0:
                print(all_losses.max(), all_losses.mean())
            if (i + 1) % val_every == 0:
                plot_val += [validate(model, batches_val)]
    return plot_train, plot_val

def validate(model, batches_val):
    model.eval()
    y_pred = []
    y_true = []
    for batch in batches_val:
        features, targets = batch
        y_true += targets.tolist()
        y_pred += model(Variable(features.view(-1, 28*28))).topk(1)[1].squeeze().data.tolist()
    model.train()
    return accuracy_score(y_true, y_pred)
