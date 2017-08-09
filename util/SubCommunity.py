
dataname = 'dblp'
subC = open('/Users/wenboxie/Data/network/dblp/com-dblp.top5000.cmty.txt', 'r')

out_nodes = open('/Users/wenboxie/Data/network/dblp/nodes.dblp_top5000', 'w')
out_nodes.write('ID,Label,Community\n')

nodeMap = {}
i=0
c=0
for line in subC:
    ns = line.split()
    for n in ns:

        if n in nodeMap.keys():
            print(n)
            continue

        out_nodes.write(str(i) + ',' + n + ',' + str(c) + '\n')
        nodeMap[n] = i
        i += 1
    c+=1
subC.close()
out_nodes.close()

edge = open('/Users/wenboxie/Data/network/dblp/out.dblp.txt', 'r')
out_edges = open('/Users/wenboxie/Data/network/dblp/out.dblp_top5000', 'w')
out_edges.write('#'+str(len(nodeMap))+'\n')
out_edges.write('#Source,Target\n')
for line in edge:
    l=line.split()
    s= l[0]
    t=l[1]
    if s in nodeMap.keys() and t in nodeMap.keys():
        out_edges.write(str(nodeMap.get(s))+','+str(nodeMap.get(t))+'\n')
edge.close()
out_edges.close()


