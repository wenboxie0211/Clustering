from numpy import *
from scipy import sparse
import pandas
#soc-advogato
dataname = 'soc-hamsterster'
# file = open('/Users/wenboxie/Data/network/'+dataname+'/original.' + dataname + '.csv')
file = open('/Users/wenboxie/Data/network/'+dataname+'/network.txt')
# file = open('/Users/wenboxie/Data/network/'+dataname+'/network.dat')

#数据映射
node_map = {}
node_set = set()
n = 0
file.readline()
for line in file.readlines():
    att = line.split(' ')
    s = att[0]
    t = att[1]
    if s not in node_set:
        node_set.add(s)
        node_map[s] = n
        n += 1
    if t not in node_set:
        node_set.add(t)
        node_map[t] = n
        n += 1
file.close()

file = open('/Users/wenboxie/Data/network/'+dataname+'/network.txt')
L = zeros((n,n))

file.readline()

while 1:
    line = file.readline()
    if not line:
        break

    att = line.split(' ')
    s = att[0]
    t = att[1]
    # L[int(node[0])-1,int(node[1])-1] = 1
    # L[int(node[1])-1,int(node[0])-1] = 1
    # L[int(float(node[0])),int(float(node[1]))] = -1
    # L[int(float(node[1])),int(float(node[0]))] = -1
    # L[int(float(node[0])),int(float(node[0]))] += 1

    #
    # L[int(node_map[s]), int(node_map[t])] = -1
    # L[int(node_map[t]), int(node_map[s])] = -1
    # L[int(node_map[s]), int(node_map[s])] += 1
    # L[int(node_map[t]), int(node_map[t])] += 1

    #
    L[int(s)-1, int(t)-1] = -1
    L[int(t)-1, int(s)-1] = -1
    L[int(s)-1, int(s)-1] += 1
    L[int(t)-1, int(t)-1] += 1

    pass  # do something
file.close()


# for i in range(0,L[0].size):
#     d = 0
#     for j in range(0,L[0].size):
#         if i == j:
#             continue
#         if L[i,j] != 0:
#             d += -L[i,j]
#             L[i,j] = -L[i,j]
#     L[i,i] = d
# print(L[0,0],L[1,1])
print('load data')

# print(ECTD(L))

L_plus = linalg.pinv(L)

print('create L_plus')

ec = zeros((n,n))
co = zeros((n,n))

VG = 0

for k in range(0,n):
    VG += L[k,k]
for i in range(0,n):
    for j in range(0,i):
        if i!=j and L[i,j]==-1:
            if (abs(L_plus[i,j])<1.0e-10):
                L_plus[i,j]=0
            # if (L_plus[i,i] + L_plus[j,j] - 2*L_plus[i,j]<0):
            #     ec[i,j] = 0.0
            else:
                # try:
            ec[i,j] = math.pow(VG * (L_plus[i,i] + L_plus[j,j] - 2*L_plus[i,j]), 0.5)
                # except:
                #     ec[i,j] = 100000
            ec[j,i] = ec[i,j]
            # co[i,j] = L_plus[i,j] / math.pow(L_plus[i,i] * L_plus[j,j], 0.5)
            # co[j,i] = co[i,j]

print('Complete!')

out_ectd = open('/Users/wenboxie/Data/network/'+dataname+'/ectd.csv', 'w')
# out_co = open('/Users/wenboxie/Data/network/'+dataname+'/cosin.' + dataname+'.csv', 'w')
# out_lplus = open('/Users/wenboxie/Data/network/'+ dataname +'/lplus.'+dataname+'.csv', 'w')
out_ectd.write('Source,Target,Weight\n')

max=0
min=10000
for i in range(0,ec[0].size):
    for j in range(0,i):
        if abs(L_plus[i,j]>10000): continue
        if ec[i,j]>max: max = ec[i,j]
        if ec[i,j]<min: min = ec[i,j]

#
for i in range(0,L[0].size):
    for j in range(0,i):
        if abs(L_plus[i,j])<=10000 and L[i,j]==-1:
            out_ectd.write(str(i)+','+str(j)+','+str(((ec[i,j]))/(max))+'\n')
            # out_ectd.write(str(i) + ',' + str(j) + ',' + str(ec[i,j]) + '\n')
            # out_co.write(str(i) + ',' + str(j) + ',' + str(co[i,j]) + '\n')
            # out_lplus.write(str(i) + ',' + str(j) + ',' + str(L_plus[i,j]) + '\n')

out_ectd.close()
# out_co.close()
# out_lplus.close()
# print(sim)
