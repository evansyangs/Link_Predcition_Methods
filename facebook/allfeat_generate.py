# 生成allfeat文件

import numpy as np
import os

DATA_DIR = './facebook/'
# 获取facebook数据文件列表
facebook_files = os.listdir(DATA_DIR)
# 寻找所有的.egofeat文件
egofeat_files = [filename for filename in facebook_files if '.egofeat' in filename]

for egofeat_file in egofeat_files:
    
    idx = egofeat_file.split('.')[0]
    
    # 在egofeat的第一位添加idx序号，再把feat添加在后面
    allfeat = np.append(np.append(int(idx),np.loadtxt(DATA_DIR+idx+'.egofeat')).reshape(1,-1),
                             np.loadtxt(DATA_DIR+idx+'.feat'),axis=0).astype(int)  # 类型为整型

    # 存储数据
    np.savetxt(DATA_DIR+idx+'.allfeat',allfeat, fmt='%i') # 保存为整型
    print(idx+'.allfeat saved!')



