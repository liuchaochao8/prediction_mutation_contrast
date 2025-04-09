#- 
import torch 
import torch.nn as nn
import torch.nn.functional as F

"""
    损失函数的意义：【一个相对的概念】
        两个数据集，一个是蛋白质的参考序列数据集，一个是基于蛋白序列的变异序列的数据集，
        其中变异序列中含有标签，标签表明变异后，蛋白质的性质的变化程度。
        要进行对比学习，如果变异后蛋白性质变化较大，那么变异序列的嵌入就应该相当于原序列较远，
        如果性质变化不大，那么就相对于原序列不远。
"""
class ProMutaContrastLoss( nn.Module ):
    def __init__(self, temperature): # 仅仅初始化温度系数
        super(ProMutaContrastLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, wild_emb, muta_emb, prob_muta, batch_size):
        # distance = F.pairwise_distance(wild_emb, muta_emb, p=2)
        # loss = torch.mean((distance - labels) ** 2)   # 暂时用均方误差替代；实际上

        # 余弦相似度-1-y/max(y)的绝对值

        wild_emb_re = wild_emb.reshape(batch_size,-1)
        muta_emb_re = muta_emb.reshape(batch_size, -1)

        # print(wild_emb_re.size(), muta_emb_re.size(), prob_muta.size())

        loss = torch.mean(torch.abs(torch.cosine_similarity(wild_emb_re, muta_emb_re) - 2 + prob_muta/torch.max(prob_muta)))
        return loss



