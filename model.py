import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from setmodules import SAB, PMA

# Reference implementation (TensorFlow):
# https://github.com/google-research/google-research/tree/master/stacked_capsule_autoencoders


def relu1(x):
    return F.relu6(x * 6.) / 6.


def safe_log(x, eps=1e-16):
    return torch.log(x.clamp(min=eps))


def geometric_transform(pose_tensor, similarity=False, nonlinear=True, as_matrix=False):
    """Converts a parameter tensor into an affine or similarity transform.
    Adapted from capsules/math_ops.py of the reference implementation.

    Args:
    pose_tensor: [..., 6] tensor.
    similarity: bool.
    nonlinear: bool; applies nonlinearities to pose params if True.
    as_matrix: bool; returns a [..., 3, 3] homogeneous matrix if True.

    Returns:
    [..., 3, 3] tensor if `as_matrix` else [..., 6] tensor.
    """
    scale_x, scale_y, theta, shear, trans_x, trans_y = torch.split(pose_tensor, 1, -1)

    if nonlinear:
        scale_x, scale_y = torch.sigmoid(scale_x) + 1e-2, torch.sigmoid(scale_y) + 1e-2
        trans_x, trans_y, shear = torch.tanh(trans_x * 5.), torch.tanh(trans_y * 5.), torch.tanh(shear * 5.)
        theta = theta * (2. * math.pi)  # not in-place: theta is a view returned by split
    else:
        scale_x, scale_y = (abs(i) + 1e-2 for i in (scale_x, scale_y))

    c, s = torch.cos(theta), torch.sin(theta)

    if similarity:
        scale = scale_x
        pose = [scale * c, -scale * s, trans_x, scale * s, scale * c, trans_y]
    else:
        pose = [
            scale_x * c + shear * scale_y * s, -scale_x * s + shear * scale_y * c,
            trans_x, scale_y * s, scale_y * c, trans_y
        ]

    pose = torch.cat(pose, -1)
    if as_matrix:
        pose = to_matrix(pose)
    return pose


def to_matrix(pose):
    """[..., 6] affine params -> [..., 3, 3] homogeneous matrix."""
    last = torch.zeros_like(pose[..., :3])
    last[..., 2] = 1.
    return torch.cat([pose, last], -1).view(*pose.shape[:-1], 3, 3)


class BatchMLP(nn.Module):
    """Applies n independent MLPs to n inputs: [B, n, d_in] -> [B, n, d_out]."""
    def __init__(self, n, dim_input, hiddens, use_bias=True):
        super(BatchMLP, self).__init__()
        dims = [dim_input] + list(hiddens)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            w = torch.empty(n, d_in, d_out)
            nn.init.trunc_normal_(w, std=1. / math.sqrt(d_in), a=-2. / math.sqrt(d_in), b=2. / math.sqrt(d_in))
            self.weights.append(nn.Parameter(w))
            is_last = i == len(dims) - 2
            self.biases.append(nn.Parameter(torch.zeros(n, d_out)) if (use_bias or not is_last) else None)

    def forward(self, x):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.einsum('bnd,nde->bne', x, w)
            if b is not None:
                x = x + b
            if i < len(self.weights) - 1:
                x = F.relu(x)
        return x


class PCAE(nn.Module):
    def __init__(self, config=None, num_capsules=24, template_size=11, num_features=16,
                 image_size=28, noise_scale=4., output_scale=1.):
        super(PCAE, self).__init__()
        self.num_capsules = num_capsules
        self.num_features = num_features
        self.image_size = image_size
        self.noise_scale = noise_scale
        self.output_scale = output_scale
        # per capsule: 6 pose + special features + 1 presence, plus 1 attention logit
        self.num_feature_maps = 6 + num_features + 1 + 1
        self.capsules = nn.Sequential(nn.Conv2d(1,128,3,stride=2),
                            nn.ReLU(),
                        nn.Conv2d(128,128,3,stride=2),
                            nn.ReLU(),
                        nn.Conv2d(128,128,3,stride=1),
                            nn.ReLU(),
                        nn.Conv2d(128,128,3,stride=1),
                            nn.ReLU(),
                        nn.Conv2d(128,num_capsules*self.num_feature_maps,1,stride=1))

        # templates are orthogonal to each other at init and scaled to [0, 1] (as in the reference)
        n_elems = template_size * template_size
        n = max(num_capsules, n_elems)
        q = np.linalg.qr(np.random.uniform(size=[n, n]))[0]
        q = q[:num_capsules, :n_elems].reshape(num_capsules, 1, template_size, template_size)
        q = (q - q.min()) / (q.max() - q.min())
        self.template_logits = nn.Parameter(torch.tensor(q, dtype=torch.float32))
        self.temperature_logit = nn.Parameter(torch.zeros(1))
        self.bg_logit = nn.Parameter(torch.zeros(1))

    @property
    def templates(self):
        # prevent negative ink
        return relu1(self.template_logits)

    def forward(self,x,mode='train'):
        B = x.shape[0]
        M = self.num_capsules
        outputs = self.capsules(x)
        outputs = outputs.view(B,M,self.num_feature_maps,-1) #(B,M,24,H*W)
        attention_soft = F.softmax(outputs[:,:,-1:,:],dim=-1)
        part_capsule_param = (outputs[:,:,:-1,:]*attention_soft).sum(-1) #(B,M,23)

        pose_param = part_capsule_param[:,:,:6]
        c_z = part_capsule_param[:,:,6:6+self.num_features]
        presence_logit = part_capsule_param[:,:,-1]
        if mode == 'train':
            presence_logit = presence_logit + (torch.rand_like(presence_logit)-.5)*self.noise_scale
        d_m = torch.sigmoid(presence_logit) #(B,M)
        x_m = geometric_transform(pose_param) #(B,M,6)

        # Affine transform. The pose maps template coordinates to image coordinates, while
        # grid_sample needs the opposite mapping, hence the inverse.
        templates = self.templates
        inv_pose = torch.linalg.inv(to_matrix(x_m))[...,:2,:] #(B,M,2,3)
        size = torch.Size((B*M,1,self.image_size,self.image_size))
        grid = F.affine_grid(inv_pose.reshape(B*M,2,3),size,align_corners=True)
        transformed_templates = F.grid_sample(templates.repeat(B,1,1,1),grid,align_corners=True)
        transformed_templates = transformed_templates.view(B,M,self.image_size,self.image_size)

        # learned constant background component which is always present
        bg_image = torch.sigmoid(self.bg_logit).expand(B,1,self.image_size,self.image_size)
        components = torch.cat([transformed_templates,bg_image],1) #(B,M+1,H,W)
        presence = torch.cat([d_m,d_m.new_ones(B,1)],1) #(B,M+1)

        # mixing probabilities are proportional to template intensity and to part presence
        temperature = F.softplus(self.temperature_logit+.5)+1e-4
        mixing_logits = components/temperature + safe_log(presence)[:,:,None,None]
        mixing_log_prob = F.log_softmax(mixing_logits,dim=1)
        component_log_prob = torch.distributions.Normal(components,self.output_scale).log_prob(x)
        log_likelihood = torch.logsumexp(component_log_prob+mixing_log_prob,dim=1).sum((-1,-2)).mean() #scalar

        # input to the OCAE: gradients are stopped for poses, presences and templates
        # but not for the special features (as in the reference)
        template_flat = templates.detach().view(1,M,-1).expand(B,-1,-1) #(B,M,11*11)
        input_ocae = torch.cat([x_m.detach(),1.-d_m.detach().unsqueeze(-1),c_z,template_flat],-1) #(B,M,144)

        return log_likelihood,input_ocae,x_m.detach(),d_m.detach()


class SetTransformer(nn.Module):
    """Set transformer configured as in the reference: a stack of self-attention
    blocks followed by attention with K learned seed vectors."""
    def __init__(self, dim_input, num_outputs, dim_output,
            num_layers=3, dim_hidden=16, num_heads=1, ln=True):
        super(SetTransformer, self).__init__()
        self.inp = nn.Linear(dim_input, dim_hidden)
        self.enc = nn.Sequential(*[SAB(dim_hidden, dim_hidden, num_heads, ln=ln) for _ in range(num_layers)])
        self.out = nn.Linear(dim_hidden, dim_output)
        self.dec = PMA(dim_output, num_heads, num_outputs)

    def forward(self, X):
        return self.dec(self.out(self.enc(self.inp(X))))


class OCAE(nn.Module):
    def __init__(self,config=None,dim_input=144,num_parts=24,num_capsules=24,set_out=256,set_head=1,
                 caps_params=32,hiddens=128,noise_scale=4.):
        super(OCAE,self).__init__()
        self.num_parts = num_parts
        self.num_capsules = num_capsules
        self.noise_scale = noise_scale
        self.set_transformer = SetTransformer(dim_input,num_capsules,set_out,num_heads=set_head)
        self.caps_param_mlp = BatchMLP(num_capsules,set_out,[hiddens,caps_params])
        # outputs: dynamic part of the capsule-part relations (OP), object-viewer relation (OV),
        # capsule presence, per-vote presence and per-vote scale
        self.splits = [num_parts*6, 6, 1, num_parts, num_parts]
        # no bias in the output layer to separate the static and the dynamic parts of OP
        self.caps_mlp = BatchMLP(num_capsules,caps_params+1,[hiddens,sum(self.splits)],use_bias=False)
        self.bias = nn.Parameter(torch.zeros(num_capsules,sum(self.splits[1:])))
        limit = math.sqrt(6. / (num_capsules*(num_parts+6)))
        self.cpr_static = nn.Parameter(torch.empty(1,num_capsules,num_parts,6).uniform_(-limit,limit))

    def forward(self,inp,x_m,d_m,mode='train'):
        B, K, M = inp.shape[0], self.num_capsules, self.num_parts
        h = self.set_transformer(inp) #(B,K,256)
        caps_params = self.caps_param_mlp(h) #(B,K,32)
        caps_params = torch.cat([caps_params,caps_params.new_ones(B,K,1)],-1)
        all_params = self.caps_mlp(caps_params)
        cpr_dynamic = all_params[:,:,:M*6].view(B,K,M,6)
        rest = all_params[:,:,M*6:]+self.bias
        ccr,pres_logit_caps,pres_logit_vote,scale_vote = torch.split(rest,self.splits[1:],-1)
        if mode == 'train':
            pres_logit_caps = pres_logit_caps + (torch.rand_like(pres_logit_caps)-.5)*self.noise_scale
            pres_logit_vote = pres_logit_vote + (torch.rand_like(pres_logit_vote)-.5)*self.noise_scale

        ov_k = geometric_transform(ccr,as_matrix=True).unsqueeze(2) #(B,K,1,3,3)
        op_kn = geometric_transform(cpr_dynamic+self.cpr_static,as_matrix=True) #(B,K,M,3,3)
        mu_kn = torch.matmul(ov_k,op_kn)[...,:2,:].reshape(B,K,M,6) #(B,K,M,6)
        a_k = torch.sigmoid(pres_logit_caps) #(B,K,1)
        a_kn = a_k*torch.sigmoid(pres_logit_vote) #(B,K,M)
        lambda_kn = F.softplus(scale_vote+.5)+1e-2 #(B,K,M) standard deviations, bounded away from 0

        # Gaussian mixture over object capsules for every part, computed in log space.
        # A dummy component with a small constant probability explains outlier parts.
        dummy = math.log(1e-2)
        vote_log_prob = torch.distributions.Normal(mu_kn,lambda_kn.unsqueeze(-1)).log_prob(x_m.unsqueeze(1)).sum(-1) #(B,K,M)
        vote_log_prob = torch.cat([vote_log_prob,vote_log_prob.new_full((B,1,M),dummy)],1)
        mixing_logits = torch.cat([safe_log(a_kn),a_kn.new_full((B,1,M),dummy)],1) #(B,K+1,M)
        posterior_logits = mixing_logits+vote_log_prob
        part_log_likelihood = torch.logsumexp(posterior_logits,1) #(B,M)
        log_likelihood = (part_log_likelihood*d_m).sum(-1).mean() #scalar

        posterior = F.softmax(posterior_logits,1)[:,:-1] #(B,K,M)
        caps_presence = a_kn.max(-1).values #(B,K)
        dynamic_l2 = 0.5*cpr_dynamic.pow(2).sum()/B
        return log_likelihood,caps_presence,posterior,dynamic_l2


class SCAE(nn.Module):
    def __init__(self,config=None,num_parts=24,num_objects=24):
        super(SCAE,self).__init__()
        self.pcae = PCAE(config,num_capsules=num_parts)
        self.ocae = OCAE(config,num_parts=num_parts,num_capsules=num_objects)

    def forward(self,x,mode='train'):
        image_likelihood,input_ocae,x_m,d_m = self.pcae(x,mode)
        part_likelihood,caps_presence,posterior,dynamic_l2 = self.ocae(input_ocae,x_m,d_m,mode)
        return image_likelihood,part_likelihood,caps_presence,posterior,dynamic_l2


class SCAE_LOSS(nn.Module):
    def __init__(self,num_classes=10,prior_within_weight=1.,prior_between_weight=1.,
                 posterior_within_weight=10.,posterior_between_weight=10.,dynamic_l2_weight=10.):
        super(SCAE_LOSS,self).__init__()
        self.num_classes = num_classes
        self.prior_within_weight = prior_within_weight
        self.prior_between_weight = prior_between_weight
        self.posterior_within_weight = posterior_within_weight
        self.posterior_between_weight = posterior_between_weight
        self.dynamic_l2_weight = dynamic_l2_weight

    @staticmethod
    def entropy(p,dim):
        p = p/(p.sum(dim,keepdim=True)+1e-8)
        return -(p*safe_log(p)).sum(dim).mean()

    def forward(self,output_scae):
        img_lik,part_lik,caps_presence,posterior,dynamic_l2 = output_scae
        B, K = caps_presence.shape
        # prior sparsity: every example should activate K/C capsules and every capsule
        # should be active in B/C examples of the batch
        prior_within = (caps_presence.sum(1)-K/self.num_classes).pow(2).mean()
        prior_between = (caps_presence.sum(0)-B/self.num_classes).pow(2).mean()
        # posterior sparsity: every example should be explained by few capsules (low entropy
        # within an example) while all capsules get used across the batch (high entropy between)
        mass = posterior.mean(-1) #(B,K)
        posterior_within = self.entropy(mass,1)
        posterior_between = self.entropy(mass.sum(0),0)
        return (-img_lik-part_lik
                +self.dynamic_l2_weight*dynamic_l2
                +self.prior_within_weight*prior_within
                +self.prior_between_weight*prior_between
                +self.posterior_within_weight*posterior_within
                -self.posterior_between_weight*posterior_between)


if __name__ == "__main__":
    scae = SCAE()
    output = scae(torch.rand(5,1,28,28))
    for out in output:
        print(out.size())
    print(SCAE_LOSS()(output))
