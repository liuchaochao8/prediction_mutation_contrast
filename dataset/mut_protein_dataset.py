
import pandas as pd

import torch
from   torch.utils.data import Dataset
from   torch.utils.data import DataLoader

#----问题：是否要对突变序列进行裁剪
#         若要裁剪，提前统一裁剪

#----突变蛋白数据集类
class MutaProDataSet(Dataset):
    def __init__(self, wild_data_path, muta_data_path ):
        """
          参考序列示例: uniprot_id  + sequence
          变异序列示例: 变异序列id   + uniprot_id + 变异序列 + label 
        """
        self.wild_df = pd.read_csv( wild_data_path , engine='python') # 参考序列数据读取
        self.muta_df = pd.read_csv( muta_data_path , engine='python') # 变异序列数据读取
        
        # 参考序列字典：uniproID-序列
        self.wild_dict = dict( zip( 
                                   self.wild_df['uniprot_id'],
                                   self.wild_df['sequence'])
                              )
        # print(next(iter(self.wild_dict.items())))
        # 原理：随机抽取变异序列，找到参考序列，计算损失，进行对比学习
    
    def __len__(self):
        return len( self.muta_df )
    
    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
            
            muta_row = self.muta_df.iloc[idx]
            muta_seq = muta_row['variant_sequence']    # 变异序列
            wild_id  = muta_row['uniprot_id']  # 参考序列id
            #label    = muta_row['label']       # 标签
            prob_muta = muta_row['am_pathogenicity']    # 致病概率

            wild_seq = self.wild_dict[wild_id] # 对应的wild sequence
            
            #----是否进行序列处理?
            #----是否进行标签处理?

            # return torch.tensor(wild_seq),\
            #        torch.tensor(muta_seq),\
            #        torch.tensor(prob_muta, dtype=torch.float32)
            return wild_seq, muta_seq, prob_muta
            
# def __init__(self, wild_data_path, muta_data_path, batch_size, valid_size)，先不包含验证集
class MutaProDataWrapper(object):
    def __init__(self, wild_data_path, muta_data_path, batch_size):
        super(object, self).__init__()
        self.batch_size            = batch_size                                      # 批数量
        self.muta_protein_data_set = MutaProDataSet(wild_data_path, muta_data_path)  # 数据加载
    
    def get_data_loader(self,dataset,batch_size):
        data_loader = DataLoader( self.muta_protein_data_set, batch_size = self.batch_size,
                                  drop_last=True, num_workers = 5)
        return data_loader
        
        
    