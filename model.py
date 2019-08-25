import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
import torchvision.transforms.functional as F1
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
from torch.nn import Parameter
from copy import deepcopy



class PCAE(nn.Module):
    def __init__(self, num_capsules, template_size, num_templates,num_feature_maps):
        super(PCAE,self).__init__()
        
        self.capsules = nn.ModuleList([nn.Sequential(nn.Conv2d(1,128,3,stride=2),
                            nn.ReLU(),
                        nn.Conv2d(128,128,3,stride=2),
                            nn.ReLU(),
                        nn.Conv2d(128,128,3,stride=1),
                            nn.ReLU(),
                        nn.Conv2d(128,128,3,stride=1),
                            nn.ReLU(),
                        nn.Conv2d(128,num_feature_maps,1,stride=1)
                            ) for _ in range(num_capsules)])
        self.templates = nn.ModuleList([ nn.Parameter(torch.randn(1,template_size,template_size))
                            for _ in range(num_templates)])
        self.soft_max = nn.Softmax(2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.to_pil = tt.ToPILImage()
        self.to_tensor = tt.ToTensor()
        

        def forward(self,x):
            outputs = [ capsule(x) for capsule in self.capsules ]
            temp= []
            for part_capsule in outputs:
                attention = part_capsule[:,-1,:,:]
                attention_soft = self.soft_max(attention.view(*attention.size()[:2],-1)).view_as(attention)
                feature_maps = part_capsule[:,:-1,:,:]
                attention_pool = torch.sum(torch.sum(feature_maps*attention_soft,dim=-1),dim=-1) #(B,6+1+16)
                temp.append(attention_pool.unsqueeze(1))
            part_capsule_param = torch.cat(temp,dim=1) #(B,M,23)
            x_m,d_m,c_z = self.relu(part_capsule_param[:,:,:6]),self.sigmoid(part_capsule_param[:,:,6]),self.relu(part_capsule_param[:,:,7:])

            # pytorch doesn't  support batch transforms 
            temp=[]
            for pose in x_m:
                temp2=[]
                for pos,template in enumerate(self.templates):
                    temp2.append(self.to_tensor(F1.resize(F1.affine(self.to_pil(template),
                                                            pose[pos][0],
                                                            (pose[pos][1],pose[pos][2]),
                                                            pose[pos][3],
                                                            (pose[pos][4],pose[pos][5])),x.size()[2:])))
                temp.append(torch.cat(temp2,0).unsqueeze(0)) #(1,M,28,28)
            transformed_templates = torch.cat(temp,0) #(B,M,28,28)
            mix_prob = self.soft_max(d_m*transformed_templates.view(*transformed_templates.size()[:2],-1)).view_as(transformed_templates)
            std= x.view(*x.size()[:2],-1).std(-1).unsqueeze(1)  #(B,1,1)
            multiplier = (std*math.pi*2).sqrt().reciprocal()  #(B,1,1)
            power_multiply = (-(2*(std**2))).reciprocal() #(B,1,1)
            detach_x = x.detach()
            gaussians = multiplier*((((detach_x-transformed_templates)**2)*power_multiply).exp()) #(B,M,28,28)
            log_likelihood = torch.sum(gaussians*mix_prob,dim=1).log().sum(-1).sum(-1).mean() #scalar loss

            return log_likelihood,c_z,x_m.detach(),d_m.detach()


















    

