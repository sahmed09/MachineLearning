import pandas as pd
# import modin.pandas as mp  # pip install modin[ray]
# import ray
import time

"""Dataset: https://www.kaggle.com/datasets/skihikingkevin/csgo-matchmaking-damage?resource=download&select=esea_master_dmg_demos.part2.csv"""

start = time.time()
df = pd.read_csv('esea_master_dmg_demos.part2.csv')
print(df)
end = time.time()
print(end - start, 's')

start = time.time()
print(df.groupby('att_team').count())
end = time.time()
print(end - start, 's')

# ray.init()
#
# start = time.time()
# df = mp.read_csv('esea_master_dmg_demos.part2.csv')
# print(df)
# end = time.time()
# print(end - start, 's')
#
# start = time.time()
# print(df.groupby('att_team').count())
# end = time.time()
# print(end - start, 's')
