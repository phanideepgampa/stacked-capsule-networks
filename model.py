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
from setmodules import *


class PCAE(nn.Module):
    def __init__(self,config, num_capsules=24, template_size=11, num_templates=24,num_feature_maps=24):
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
        self.templates = [ nn.Parameter(torch.randn(1,template_size,template_size))
                            for _ in range(num_templates)]
        self.soft_max = nn.Softmax(2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.to_pil = tt.ToPILImage()
        self.to_tensor = tt.ToTensor()
        
    def forward(self,x,device,mode='train'):
        outputs = [ capsule(x) for capsule in self.capsules ]
        temp= []
        for part_capsule in outputs:
            attention = part_capsule[:,-1,:,:].unsqueeze(1)
            attention_soft = self.soft_max(attention.view(*attention.size()[:2],-1)).view_as(attention)
            feature_maps = part_capsule[:,:-1,:,:]
            attention_pool = torch.sum(torch.sum(feature_maps*attention_soft,dim=-1),dim=-1) #(B,6+1+16)
            temp.append(attention_pool.unsqueeze(1))
        part_capsule_param = torch.cat(temp,dim=1) #(B,M,23)
        if mode == 'train':
            noise_1 = torch.FloatTensor(*part_capsule_param.size()[:2]).uniform_(-2,2).to(device)
        else:
            noise_1 = torch.zeros(*part_capsule_param.size()[:2]).to(device)
        x_m,d_m,c_z = self.relu(part_capsule_param[:,:,:6]),self.sigmoid(part_capsule_param[:,:,6]+noise_1).view(*part_capsule_param.size()[:2],1),self.relu(part_capsule_param[:,:,7:])

        epsilon = torch.tensor(1e-6).to(device)
        # pytorch doesn't  support batch transforms 
        temp=[]
        for pose in x_m:
            temp2=[]
            for pos,template in enumerate(self.templates):
                temp2.append(self.to_tensor(F1.resize(F1.affine(self.to_pil(template),
                                                        pose[pos][0],
                                                        (pose[pos][1],pose[pos][2]),
                                                        pose[pos][3]+epsilon, # for accounting zero scale values
                                                        (pose[pos][4],pose[pos][5])),x.size()[2:])))
            temp.append(torch.cat(temp2,0).unsqueeze(0)) #(1,M,28,28)
        transformed_templates = torch.cat(temp,0) #(B,M,28,28)
        mix_prob = self.soft_max(d_m*transformed_templates.view(*transformed_templates.size()[:2],-1)).view_as(transformed_templates)
        std= x.view(*x.size()[:2],-1).std(-1).unsqueeze(1)  #(B,1,1)
        multiplier = (std*math.pi*2).sqrt().reciprocal().unsqueeze(-1)  #(B,1,1,1)
        power_multiply = (-(2*(std**2))).reciprocal().unsqueeze(-1) #(B,1,1,1)
        detach_x = x.data
        gaussians = multiplier*((((detach_x-transformed_templates)**2)*power_multiply).exp()) #(B,M,28,28)
        log_likelihood = torch.sum(gaussians*mix_prob,dim=1).log().sum(-1).sum(-1).mean() #scalar loss
        x_m_detach = x_m.data
        d_m_detach = d_m.data
        template_det = []
        for template in self.templates:
            template_det.append(template.data.view(1,-1))
        template_detached = torch.cat(template_det,0).unsqueeze(0).expand(x_m_detach.shape[0],-1,-1) #(B,M,11*11)
        input_ocae = torch.cat([d_m_detach,x_m_detach,template_detached,c_z],-1) #(B,M,144)
        
        return log_likelihood,input_ocae,x_m_detach,d_m_detach


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))

class OCAE(nn.Module):
    def __init__(self,config,dim_input=144,num_capsules=24,set_out=256,set_head=1,special_feat=16):
        super(OCAE,self).__init__()

        self.set_transformer = nn.Sequential( SetTransformer(dim_input,num_capsules,set_out,num_heads=set_head,dim_hidden=16,ln=True),
                                              SetTransformer(set_out,num_capsules,set_out,num_heads=set_head,dim_hidden=16,ln=True),
                                              SetTransformer(set_out,num_capsules,special_feat+1+9,num_heads=set_head,dim_hidden=16,ln=True),
                                                         )
        self.mlps = nn.ModuleList( [ nn.Sequential( nn.Linear(special_feat,special_feat),
                                                     nn.ReLU(),
                                                     nn.Linear(special_feat,48)) for _ in range(num_capsules) ] )
        self.op_mat = Parameter(torch.randn(num_capsules,num_capsules,3,3))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,inp,x_m,d_m,device,mode='train'):
        object_parts = self.set_transformer(inp) #(B,K,9+16+1)
        if mode == 'train':
            noise_1 = torch.FloatTensor(*object_parts.size()[:2]).uniform_(-2,2).to(device)
            noise_2 = torch.FloatTensor(object_parts.shape[0],24).uniform_(-2,2).to(device)

        else:
            noise_1 = torch.zeros(*object_parts.size()[:2]).to(device)
            noise_2 = torch.zeros(object_parts.shape[0],24).to(device)

        ov_k,c_k,a_k = self.relu(object_parts[:,:,:9]).view(*object_parts.size()[:2],1,3,3),self.relu(object_parts[:,:,9:25]),self.sigmoid(object_parts[:,:,-1]+noise_1).view(*object_parts.size()[:2],1,1,1)        
        temp_a =[]
        temp_lambda = []
        for num,mlp in enumerate(self.mlps):
            mlp_out = self.mlps[num](c_k[:,num,:])
            temp_a.append(self.sigmoid(mlp_out[:,:24]+noise_2).unsqueeze(1))
            temp_lambda.append(self.relu(mlp_out[:,24:]).unsqueeze(1))
        a_kn = torch.cat(temp_a,1).unsqueeze(-1).unsqueeze(-1) #(B,K,M,1,1)
        lambda_kn = torch.cat(temp_lambda,1).unsqueeze(-1).unsqueeze(-1) #(B,K,M,1,1)
        v_kn = ov_k.matmul(self.op_mat) #(B,K,M,3,3)
        mu_kn = v_kn.view(*v_kn.size()[:3],-1)[:,:,:,:6] #(B,K,M,6)
        x_m = x_m.unsqueeze(1) #(B,1,M,6)
        diff = (x_m - mu_kn).unsqueeze(-2) #(B,K,M,1,6)
        identity = torch.eye(6).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(*diff.size()[:3],-1,-1).to(device) #(B,K,M,6,6)
        cov_matrix = lambda_kn*identity #(B,K,M,6,6)
        mahalanobis = torch.matmul(torch.matmul(diff,cov_matrix.reciprocal()),diff.transpose(-1,-2)) #(B,K,M,1,1)
        gaussian_multiplier = (((2*math.pi)**6)*(lambda_kn**6)).sqrt() #(B,K,M,1,1)
        gaussian = (-0.5*mahalanobis).exp()*gaussian_multiplier.reciprocal() #(B,K,M,1,1)

        gaussian_component = (a_k*a_kn)*((a_k.sum(1).unsqueeze(1)*a_kn.sum(2).unsqueeze(1)).reciprocal()) #(B,K,M,1,1)

        gauss_mix = (gaussian*gaussian_component).squeeze(-1).squeeze(-1) #(B,K,M)

        before_log = gauss_mix.sum(1).log() #(B,M)
        log_likelihood = (before_log*(d_m.view(before_log.shape[0],-1))).sum(-1).mean() #scalar
        return log_likelihood, a_k,a_kn,gaussian

class SCAE(nn.Module):
    def __init__(self,config=None):
        super(SCAE,self).__init__()
        self.pcae = PCAE(config)
        self.ocae = OCAE(config)
    def forward(self,x,device,mode):
        image_likelihood,input_ocae,x_m,d_m = self.pcae(x,device,mode)
        part_likelihood,a_k,a_kn,gaussian = self.ocae(input_ocae,x_m,d_m,device,mode)
        return image_likelihood,part_likelihood,a_k,a_kn,gaussian

        


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcae = PCAE()
    temp = torch.randn(5,1,28,28).to(device)
    pcae.to(device)
    output = pcae(temp,device)
    for out in output:
        print(out.size())
    ocae = OCAE()
    output2 = ocae(output[1],output[2],output[3],device)
    for out in output2:
        print(out.size())
















    

