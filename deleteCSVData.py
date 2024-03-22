import pandas as pd
import glob
 
csv_path = './dataset/coords_dataset.csv'
df = pd.read_csv(csv_path, encoding="gbk")
data = input("输入删除行的类名: ")
# 按条件筛选待删除的行索引，以下三行脚本分别是单条件，双条件取交集，双条件取并集
row_indexs = df[df['class']== data].index
# row_indexs = df[(df["model-index"]%5000!=0) & (df["loss_pixel"]<0.5)].index
# row_indexs = df[(df["model-index"]%5000!=0) | (df["loss_pixel"]<0.5)].index
 
# 执行删除
df.drop(row_indexs,inplace=True)
df.to_csv(csv_path,index=False,encoding="gbk") 
print(row_indexs)



