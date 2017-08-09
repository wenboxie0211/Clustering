from numpy import *

file = open('/Users/wenboxie/Documents/Manuscript/normalization/data/unit2parts-matrix.csv')
out = open('/Users/wenboxie/Documents/Manuscript/normalization/data/unit2parts-edge.csv', 'w')
out.write('Source,Target\n')

line=file.readline()

num_m=0;
aij=0
while 1:
    line = file.readline().strip('\n')
    if not line:
        break

    node = line.split(',')

    print('len(node)=',len(node))
    print('node=', node)
    for i in range(1,len(node)):

        # print('n[',i,']=',node[i])
        if node[i]=='1':
            print(node[i])
            out.write('U'+str(num_m)+',P'+str(i-1)+'\n')
            # print('CE'+str(num_m)+',EQ'+str(i-1)+'\n')
            aij+=1
    num_m+=1

    pass  # do something
file.close()
out.close()
print('aij=',aij)