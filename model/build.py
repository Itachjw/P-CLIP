from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
from datasets.read_ulab import read_ulab_image
import pdb

class CrossAtt(nn.Module):
    def __init__(self,
                 embed_dim=512):
        super(CrossAtt, self).__init__()
        self.embed_dim = embed_dim
        self.ln_pre_t = LayerNorm(self.embed_dim)
        self.ln_pre_i = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)
    def forward(self, x, y):
        xx = self.cross_attn(
                self.ln_pre_t(x),
                self.ln_pre_i(y),
                self.ln_pre_i(y),
                need_weights=False)[0]
        return xx

class DiscrepLearning(nn.Module):
    def __init__(self, args, embed_dim):
        super(DiscrepLearning, self).__init__()
        
    def forward(self, x, y):
    
        x_norm = x / x.norm(dim=1, keepdim=True)
        y_norm = y / y.norm(dim=1, keepdim=True)
        
        y2x_sim = y_norm @ x_norm.transpose(2,1)
        r_sim = 1. - F.softmax(y2x_sim, dim=-1)
        feats = r_sim @ x #+ x
        
        return feats

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        #self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        self.CAtt = CrossAtt(self.embed_dim)
        self.DPL = DiscrepLearning(self.args, self.embed_dim)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _ = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch, epoch, k_uids, ulab_id_path):
        ret = dict()

        images = batch['images']
        images1 = batch['images1']
        images2 = batch['images2']

        cap_ids = batch['caption_ids']
        caption_ids1 = batch['mlm_ids1']
        caption_ids2 = batch['mlm_ids2']
        
        pids = batch['pids']
        p_uids = k_uids[pids]
        images1, images2, images3, images4, images5 = read_ulab_image(self.args, images1, images2, images3, images4, images5, p_uids, ulab_id_path)
        #pdb.set_trace()
        
        index = torch.minimum(torch.ones_like(cap_ids.argmax(dim=-1))*76, cap_ids.argmax(dim=-1)+1)
        image_feats, image_m, text_feats, text_m = self.base_model(images, cap_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), index].float()

        image_feats1, _ = self.base_model.encode_image(images1)
        i_feats1 = image_feats1[:, 0, :].float()
        text_feats1, _ = self.base_model.encode_text(cap_ids)#caption_ids1)
        t_feats1 = text_feats1[torch.arange(text_feats1.shape[0]), index].float()
        
        image_feats2, _ = self.base_model.encode_image(images2)
        i_feats2 = image_feats2[:, 0, :].float()
        text_feats2, _ = self.base_model.encode_text(cap_ids)#caption_ids2)
        t_feats2 = text_feats2[torch.arange(text_feats2.shape[0]), index].float()
        
        #pdb.set_trace()

        logit_scale = 1 / 0.02 #self.logit_scale
            
        ret.update({'temperature': 1 / logit_scale})

        #pdb.set_trace()
        if 'itc' in self.current_task:

            itc_loss = objectives.compute_itc(i_feats, t_feats, logit_scale)
            loss = itc_loss
            ret.update({'g_sdm_loss':loss})
            
        if 'ccm' in self.current_task:

            
            f_loss00, matrix00, cosine00 = objectives.ccm_loss(i_feats, t_feats, batch['pids'], logit_scale)
            f_loss10, matrix10, cosine10 = objectives.ccm_loss(i_feats1, t_feats1, batch['pids'], logit_scale)
            f_loss20, matrix20, cosine20 = objectives.ccm_loss(i_feats2, t_feats2, batch['pids'], logit_scale)
              
            i_f = torch.cat((i_feats, i_feats1, i_feats2), 0)
            t_f = torch.cat((t_feats, t_feats1, t_feats2), 0)
            tal_loss = objectives.compute_TAL_per(i_f, t_f, rep=3, batch_size=i_feats1.shape[0], tau=0.02, margin=0.5)
            
            
            loss = f_loss00 + f_loss10 + f_loss20 + tal_loss
            
            ret.update({'f_sdm_loss':loss})
            
        if 'cdl' in self.current_task:

            cross_feats = self.CAtt(text_feats, image_feats)
            cross_feats1 = self.CAtt(text_feats1, image_feats1)
            cross_feats2 = self.CAtt(text_feats2, image_feats2)
            
            margin = 0.2
            att_words1 = self.DPL(cross_feats, cross_feats1)
            att_words2 = self.DPL(cross_feats, cross_feats2) 
              
            
            text_fs1, _ = self.base_model.decode_text(batch['mlm_ids1'], att_words1, batch['mlm_labels1'])
            t_fs1 = text_fs1[torch.arange(text_feats.shape[0]), index].float()
            loss1 = objectives.triplet_hard_loss(t_fs1, i_feats.detach(), t_feats1.detach(), margin)
            
           
            text_fs2, _ = self.base_model.decode_text(batch['mlm_ids2'], att_words2, batch['mlm_labels2'])
            t_fs2 = text_fs2[torch.arange(text_feats.shape[0]), index].float()
            loss2 = objectives.triplet_hard_loss(t_fs2, i_feats.detach(), t_feats2.detach(), margin)
            
               
            loss = loss1 + loss2
            ret.update({'mlm_loss': loss})

def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model