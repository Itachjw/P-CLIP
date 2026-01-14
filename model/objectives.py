import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import pdb



def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss

def ccm_loss(image_fetures, text_fetures, pid, logit_scale, epsilon=1e-8):
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    sim_matrix = F.softmax(logit_scale*t2i_cosine_theta, dim=1)*F.softmax(logit_scale*t2i_cosine_theta, dim=0)

    loss_i = F.cross_entropy(sim_matrix, labels)
    loss_t =F.cross_entropy(sim_matrix.t(), labels)
    loss = (loss_i +  loss_t)/2

    return loss, sim_matrix, t2i_cosine_theta

def hard_loss(sim1, sim2, sim3, pid, logit_scale):
    
    batch_size = sim1.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    pos = torch.min(torch.min(sim1, sim2), sim3)
    neg = torch.max(torch.max(sim1, sim2), sim3)

    #pdb.set_trace()
   
    t2i_cosine_theta = labels*pos + (1-labels)*neg
    #sim_matrix = F.softmax(logit_scale*t2i_cosine_theta, dim=1)*F.softmax(logit_scale*t2i_cosine_theta, dim=0)

    loss_i = F.cross_entropy(t2i_cosine_theta, labels)
    loss_t =F.cross_entropy(t2i_cosine_theta.t(), labels)
    loss = (loss_i +  loss_t)/2 
   
    return loss


def compute_TAL_per(i_feats, t_feats, rep, batch_size, tau=0.02, margin=0.5):

    pid = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    pid = pid.to(i_feats.device)
    pid = pid.repeat(rep)
    #pdb.set_trace()
    pid = pid.reshape((batch_size*rep, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    i_norm = i_feats / i_feats.norm(dim=-1, keepdim=True)
    t_norm = t_feats / t_feats.norm(dim=-1, keepdim=True)
    scores = t_norm @ i_norm.t()

    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    return loss.mean() 