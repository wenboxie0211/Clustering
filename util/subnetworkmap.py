
dataname = 'netsci-379'

edgesfile = open('/Users/wenboxie/Data/network/'+dataname+'/out.' + dataname + '.csv')
nodesfile = open('/Users/wenboxie/Data/network/'+dataname+'/'+dataname+'[Nodes].csv')

n=379

nodesfile.readline()
n_map = {}
i=0
for line in nodesfile:
    line.strip()
    a = line.split(',')
    n_map[a[0]] = i
    print('n_map[',a[0],']=',i)
    i+=1
nodesfile.close()

out_ori = open('/Users/wenboxie/Data/network/'+dataname+'/original.' + dataname + '.csv', 'w')
edgesfile.readline()
for line in edgesfile:
    line.strip('\n')
    a=line.split(',')
    s=a[0]
    t=a[1]
    w=a[2]
    out_ori.write(str(n_map[s]) + ',' + str(n_map[t]) + ',' + str(w))

out_ori.close()
edgesfile.close()



