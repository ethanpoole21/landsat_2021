import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

headers = ['Imname','xcoordinate1','ycoordinate1','meanradiance','stdevradiance','shiest','time']

df = pd.read_csv('1615_out/roi1_averages.csv', names=headers,delimiter=',')

a = df['ycoordinate1'].values
a = (a[1:])
a = a.astype(np.float)
b = df['meanradiance'].values
b = (b[1:])
b = b.astype(np.float)

print(a)
print(b)

plt.plot(a)
plt.title('meanradiance')
plt.savefig('1615_out/roi1_meanrad.png')
plt.show()

plt.plot(b)
plt.title('stdevradiance')
plt.savefig('1615_out/roi1_astdevrad.png')
plt.show()