from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
import pdb

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices
    
class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        #pdb.set_trace()
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1)#.cpu() # text features
        gfeats = F.normalize(gfeats, p=2, dim=1)#.cpu() # image features

        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return t2i_cmc[0]

class GeneratorLabel():
    def __init__(self, args, unlab_img_loader, ptrain_img_loader, ptrain_txt_loader, unlab_id_path):
        self.pimg_loader = ptrain_img_loader # gallery
        self.ptxt_loader = ptrain_txt_loader # query
        self.unlab_loader = unlab_img_loader
        #self.logger = logging.getLogger("IRRA.eval")
        self.ulab_id_path = unlab_id_path[0][1]
        self.confidence = args.confidence + 1e-6

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats, ufeats, uids = [], [], [], [], [], []
        # text
        for pid, caption in self.ptxt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            pidd = np.array(pid)
            qids.append(pidd)
            qfeats.append(text_feat)
        txt_ids = np.concatenate(qids, 0).astype(np.int16)
        txt_feats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.pimg_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            pidd = np.array(pid)
            gids.append(pidd)
            gfeats.append(img_feat)
        img_ids = np.concatenate(gids, 0).astype(np.int16)
        img_feats = torch.cat(gfeats, 0)

        for pid, img in self.unlab_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            pidd = np.array(pid)
            uids.append(pidd)
            ufeats.append(img_feat)
        ulab_ids = np.concatenate(uids, 0).astype(np.int16)
        ulab_feats = torch.cat(ufeats, 0)

        return txt_feats, img_feats, txt_ids, img_ids, ulab_feats, ulab_ids
    
    def eval(self, model): 

        txt_feats, img_feats, txt_ids, img_ids, ulab_feats, ulab_ids = self._compute_embedding(model)

        txt_feats = F.normalize(txt_feats, p=2, dim=1) # text features
        img_feats = F.normalize(img_feats, p=2, dim=1) # image features
        ulab_feats = F.normalize(ulab_feats, p=2, dim=1) # ulabel image features

        #t_u_dist = txt_feats @ ulab_feats.t()
        #i_u_dist = img_feats.cpu() @ ulab_feats.cpu().t()
        #u_u_dist = ulab_feats @ ulab_feats.t()
        
        #img_feats = txt_feats
        q_g_dist = img_feats.cpu() @ ulab_feats.cpu().t()
        q_q_dist = img_feats.cpu() @ img_feats.cpu().t()
        g_g_dist = ulab_feats.cpu() @ ulab_feats.cpu().t()
        
        i_u_euc = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3)
        #pdb.set_trace()

        i_u_euc = torch.from_numpy(i_u_euc)
        #t_u_euc = 2. - 2 * t_u_dist
        #i_u_euc = 2. - 2 * i_u_dist
        values, indices = torch.sort(i_u_euc, axis=-1) #np.argsort(i_u_euc.cpu().numpy(), axis=-1)
        mask = values<self.confidence
        indices = mask*indices
        #pdb.set_trace()

        num = mask.sum(dim=-1, keepdim=True)
        k_uids = indices[:,:10]
        k_uids = torch.cat((k_uids, num), dim=-1)
        
        return k_uids.cuda()


class PseudoLabel():
    def __init__(self, unlab_img_loader, ptrain_img_loader, ptrain_txt_loader, unlab_id_path):
        self.pimg_loader = ptrain_img_loader # gallery
        self.ptxt_loader = ptrain_txt_loader # query
        self.unlab_loader = unlab_img_loader
        #self.logger = logging.getLogger("IRRA.eval")
        self.ulab_id_path = unlab_id_path[0][1]

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats, ufeats, uids = [], [], [], [], [], []
        # text
        for pid, caption in self.ptxt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            pidd = np.array(pid)
            qids.append(pidd)
            qfeats.append(text_feat)
        txt_ids = np.concatenate(qids, 0).astype(np.int16)
        txt_feats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.pimg_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            pidd = np.array(pid)
            gids.append(pidd)
            gfeats.append(img_feat)
        img_ids = np.concatenate(gids, 0).astype(np.int16)
        img_feats = torch.cat(gfeats, 0)

        for pid, img in self.unlab_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            pidd = np.array(pid)
            uids.append(pidd)
            ufeats.append(img_feat)
        ulab_ids = np.concatenate(uids, 0).astype(np.int16)
        ulab_feats = torch.cat(ufeats, 0)

        return txt_feats, img_feats, txt_ids, img_ids, ulab_feats, ulab_ids
    
    def eval(self, model, k, i2t_metric=False):

        txt_feats, img_feats, txt_ids, img_ids, ulab_feats, ulab_ids = self._compute_embedding(model)

        txt_feats = F.normalize(txt_feats, p=2, dim=1) # text features
        img_feats = F.normalize(img_feats, p=2, dim=1) # image features
        ulab_feats = F.normalize(ulab_feats, p=2, dim=1) # ulabel image features

        t_u_dist = txt_feats @ ulab_feats.t()
        i_u_dist = img_feats @ ulab_feats.t()
        #u_u_dist = ulab_feats @ ulab_feats.t()

        t_u_euc = 2. - 2 * t_u_dist
        i_u_euc = 2. - 2 * i_u_dist
        
        #pdb.set_trace()
        tu_rank = np.argsort(t_u_euc.cpu().numpy(), axis=-1)[:,::-1]
        iu_rank = np.argsort(i_u_euc.cpu().numpy(), axis=-1)[:,::-1]
        
        k_uids = self.count_k_recip(tu_rank, iu_rank, k, img_ids)
        
        return k_uids

    def count_k_recip(self, tu_rank, iu_rank, k, img_ids):
        tu_rank = torch.from_numpy(tu_rank.copy()).cuda()
        iu_rank = torch.from_numpy(iu_rank.copy()).cuda()
        num = tu_rank.shape[0]
        for i in range(num):
            k_recip_uids = self.tensor_intersect(iu_rank[i,:k], tu_rank[i, :k])
            if k_recip_uids.shape[0]==0:
                k_recip_uids = tu_rank[i,:3]
            elif k_recip_uids.shape[0]==1:
                k_recip_uids = torch.cat((k_recip_uids, tu_rank[i,:2]), 0)
            elif k_recip_uids.shape[0]==2:
                k_recip_uids = torch.cat((k_recip_uids, tu_rank[i,:1]), 0)
            else:
                k_recip_uids = k_recip_uids[:3]
            if i==0:
                k_uids = k_recip_uids.unsqueeze(0)
            else:
                k_uids = torch.cat((k_uids, k_recip_uids.unsqueeze(0)), 0)
        return k_uids
         
    def tensor_intersect(self, t1, t2):
        # t1=t1.cuda()
        # t2=t2.cuda()
        indices = torch.zeros_like(t1, dtype = torch.bool, device = 'cuda')
        for elem in t2:
            indices = indices | (t1 == elem)  
            intersection = t1[indices]  
        return intersection

def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1] #第i个图片的前20个相似图片的索引号
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0] #返回backward_k_neigh_index中等于i的图片的行索引号
    return forward_k_neigh_index[fi]  #返回与第i张图片 互相为k_reciprocal_neigh的图片索引号
 
def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    q_g_dist = q_g_dist.cpu().numpy()
    q_q_dist = q_q_dist.cpu().numpy()
    g_g_dist = g_g_dist.cpu().numpy()
    #pdb.set_trace()
    
    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    
    original_dist = 2. - 2 * original_dist   #np.power(original_dist, 2).astype(np.float32) 余弦距离转欧式距离
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0)) #归一化
    V = np.zeros_like(original_dist).astype(np.float32)
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition( original_dist, range(1,k1+1) ) #取前20，返回索引号
 
    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]
 
    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh( initial_rank, i, k1) #取出互相是前20的
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)): #遍历与第i张图片互相是前20的每张图片
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
        #增广k_reciprocal_neigh数据，形成k_reciprocal_expansion_index
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index) #避免重复，并从小到大排序
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index]) #第i张图片与其前20+图片的权重
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight) #V记录第i个对其前20+个近邻的权重，其中有0有非0，非0表示没权重的，就似乎非前20+的
 
    original_dist = original_dist[:query_num,] #original_dist裁剪到 只有query x query+g
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num): #遍历所有图片
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)#第i张图片在initial_rank前k2的序号的权重平均值
                                                                #第i张图的initial_rank前k2的图片对应全部图的权重平均值
                                                                #若V_qe中(i,j)=0，则表明i的前k2个相似图都与j不相似
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])
 
    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)
 
    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)
 
    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist