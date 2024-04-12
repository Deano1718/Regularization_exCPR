import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import datetime
from datetime import datetime
import numpy as np



class Prototype(nn.Module):
    def __init__(self, num_classes, num_features, lr_proto=0.1):
        super(Prototype, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.lr_proto = lr_proto
        self.prototypes = torch.rand( [self.num_classes,self.num_features], dtype=torch.float, requires_grad=True)

    def step(self):
        latent_gradients = self.lr_proto*self.prototypes.grad
        self.prototypes.data = self.prototypes.data - latent_gradients.data
        #self.prototypes.clamp_(0.0,1e6)   #optional gaurantee to strictly enforce non-negativity
        self.prototypes.grad.zero_()


class exCPRLoss(nn.Module):
    def __init__(self, PrototypeObject, alpha=1.0, beta=0.2, gamma=3.0e5, zeta=0.2, r=20, nu=-1, use_cuda=1, verbose=0):
        super(exCPRLoss, self).__init__()

        #Hyperparameters
        self.alpha = alpha   #1.0
        self.beta = beta     #0.02
        self.gamma = gamma   #0.25
        self.zeta = zeta     #0.25
        self.r = r
        self.nu = nu

        #Object Reference to prototypes
        self.prototypes = PrototypeObject.prototypes
        self.J = PrototypeObject.num_features
        self.K = PrototypeObject.num_classes

        #verbose
        #will return loss sub-component values for tracking purposes
        self.verbose = verbose

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.prototype_warm = 0

    def forward(self, feature_vec, y_pred, y_targ):

        bs = feature_vec.size(0)

        xent_loss = self.alpha*F.cross_entropy(y_pred, y_targ)
        loss = xent_loss


        #compute unit vectors
        feature_vec_unit = F.normalize(feature_vec, dim=1)

        #select prototypes for current batch
        #move only current prototypes to GPU, Pytorch keeps track of devices for each tensor
        selections = y_targ.clone().detach().cpu()
        prototypes_cur = self.prototypes[selections].to(self.device)
        prototypes_cur_unit = F.normalize(prototypes_cur, dim=1)
        #print (prototypes_cur.shape)


        #Calculate L_proto

        #n = feature_vec.size(0)
        #m = y.size(0)
        #d = feature_vec.size(1)
        #fvx = feature_vec_unit.unsqueeze(1).expand(n, n, self.J)        #[batch, batch, J]
        #pvy = prototypes_cur_unit.unsqueeze(0).expand(n, n, self.J)     #[batch, batch, J]

        #loss_proto = self.beta*torch.pow(feature_vec_unit - prototypes_cur_unit, 2).sum(dim=1).mean()                  #[]
        if self.prototype_warm:
            loss_proto = self.beta*torch.pow(feature_vec_unit - prototypes_cur_unit, 2).sum(dim=1).mean()
            loss += loss_proto
        else:
            loss_proto = 0.0
            prototypes_cur.data = feature_vec.clone().detach().data
            
        loss += loss_proto


        #Calculate L_CS
        #Only update prototypes appearing in current batch for more efficient implementation - paper shows generic update to all prototypes

        CS_mat = prototypes_cur_unit @ prototypes_cur_unit.t()
        CS_mat_2 = CS_mat**2.0
        loss_cs = self.zeta*torch.mean(CS_mat_2.masked_select(~torch.eye(bs, dtype=bool, device=self.device)).view(bs,bs-1))
        loss += loss_cs

        #Calculate L_cov

        #clone/detach prototypes for L_cov only and sort by cur prototype values
        proto_unit_sort, proto_unit_sort_ind = torch.sort(prototypes_cur_unit.clone().detach(), dim=1)

        #reindex feature vectors
        feature_vec_unit_sorted = feature_vec_unit[torch.arange(bs).unsqueeze(1),proto_unit_sort_ind]

        #calculate cov contribution unshifted
        diffs_unshifted = proto_unit_sort*(feature_vec_unit_sorted - proto_unit_sort)

        #shift
        choice = np.random.randint(1,self.r + 1)
        diffs_shift_left = F.pad(diffs_unshifted,(0,choice))
        diffs_shift_right = F.pad(diffs_unshifted,(choice,0))


        #Many possible choices for targeting specific off-diagonal covariance contributions
        covsign = self.nu

        if covsign == 0:
            loss_cov = self.gamma*torch.mean(torch.abs(diffs_shift_left*diffs_shift_right))                #minimize the magnitude of both sign covariance contributions
        elif covsign == 1:
            loss_cov = self.gamma*torch.mean(F.relu(diffs_shift_left*diffs_shift_right))                   #minimize only the positive covariance contributions
        elif covsign == -1:
            loss_cov = self.gamma*torch.mean(F.relu(covsign*(diffs_shift_left*diffs_shift_right)))         #minimize the magnitude of the negative contributions
        elif covsign == 2:
            loss_cov = self.gamma*torch.mean(diffs_shift_left*diffs_shift_right)                           #push everything towards negative (negative conts get larger)
        elif covsign == 3:
            loss_cov = self.gamma*torch.mean(-1.0*(diffs_shift_left*diffs_shift_right))                    #push everything towards positive (positive conts get larger)
        else:
            raise Exception("Select a valid value for nu covariance parameter from: [-1,0,1,2,3]") 

        loss += loss_cov

        
        if self.verbose:
            return loss, xent_loss, loss_proto, loss_cov, loss_cs
        else:
            return loss
