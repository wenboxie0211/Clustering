import numpy as np
from util.Node import node
import random
import math

class node(object):
    #
    ## set parameters
    def __init__(self,id):
        self.id = id    # id
        self.flag = 0      # flag: root - 1 or leave - 0
        self.height = 0      # height
        self.children_set = set()
        self.sup_l = None
        self.sup_r = None

    def set_flag(self,f):
        self.flag=f

    def set_height(self,h):
        self.height=h

    def set_supporting_nodes(self,node_l,node_r):
        self.sup_l=node_l
        self.sup_r=node_r

    def add_child(self,c):
        self.children_set.add(c)

    def remove_child(self,c):
        self.children_set.remove(c)

    #
    ## get parameters
    def get_id(self):
        return self.id

    def get_chilren(self):
        return self.children_set

    def get_height(self):
        return self.height

    def get_flag(self):
        return self.flag

    def get_supporting_nodes(self):
        return self.sup_l, self.sup_r

def get_nearest_neighbors(sim):
    nn_directory = {};
    for n1 in sim:
        neighbors = sim[n1]
        mind = float('inf')
        nn = set()
        for n2 in neighbors:
            if n1 == n2:
                continue
            d = neighbors[n2]
            if d < mind:
                nn = set()
                nn.append(n2)
                mind = d
            elif d == mind:
                nn.append(n2)
        nn_directory[n1] = nn
    return nn_directory

def get_height_threshold(n, alpha):
    return  int(math.pow(n, alpha))

def get_roots(sim,nr,alpha):

    # a) Put all the nodes into a candidate set;
    nn = get_nearest_neighbors(sim)
    candidates = sim.keys()
    new_roots_set = set()

    # b) If the candidate set is not empty, choose a node from the
	# candidate set randomly, and add it into a List L as the starting
	# node, otherwise finish this clusteringlevel ;
    while len(candidates)>0:
        c = random.choice(candidates)
        link = [c]
        while True:

            # c) Search the nearest node of C and regard it as its parent
			# node P;
            p = random.choice(nn[c])

            # d) If P has already been in the list L, create a new node R
			# which is the centroid of C and P. Treat L as a tree with root
			# R and return to step b) ;
            if p in link:
                node_name = 'root.'+nr
                nr+=1
                r = node(node_name)
                r.set_flag(1)
                r.set_supporting_nodes(c,p)
                c.height = r.height + 1
                p.height = r.height + 1
                new_roots_set.add(r)
                for ele in link:
                    candidates.remove(ele)
                break

            # e) If P is a node of a tree T, join L into T as a sub-tree.
			# Due to the limitation of tree height, some nodes in the top
			# of the list L should be removed and treat each of them as a
			# new root. Return to step b);
            if p not in link:
                c.height = p.height + 1
                p.chilren_set.add(c)
                for ele in link:
                    candidates.remove(ele)
                break

            # f) Regard P as C and return to step c).
            p.chilren_set.add(c)
            link.append(c)

    # h) update similarity dictionary
    new_sim = {}
    for new_r in new_roots_set:
        l1, r1 = new_r.get_supporting_nodes()
        ab = sim[l1][r1]

        for another_r in new_roots_set:
            if new_r is another_r: continue

            l2, r2 = another_r.get_supporting_nodes()

            cd = sim[l2][r2]
            ad = sim[l1][l2]
            bc = sim[r1][r2]
            ac = sim[l1][r2]
            bd = sim[r1][l2]

            ef = math.pow((ad * ad + bc * bc + ac * ac +
                           bd * bd - ab * ab - cd * cd) / 4, 0.5)

            new_sim[new_r][another_r] = ef

    # g) when the clustering level is end, cut the leaves whose height is
	# out of the limitation

    out = set()
    for new_r in new_roots_set:
        relative_nodes = get_relative_nodes(new_r)
        height_threshold = get_height_threshold(len(relative_nodes),alpha)
        out_new_r = get_out_nodes(new_r, height_threshold)
        out.union(out_new_r)

    for out_node in out:
        for another_r in new_roots_set:
            l2, r2 = another_r.get_supporting_nodes()

            ab = sim[out_node][l2]
            ac = sim[out_node][r2]
            ab = sim[l2][r2]
            ad = math.pow((ab * ab + ac * ac)/2 - (bc * bc)/4, 0.5)

            new_sim[out_node][another_r] = new_sim[another_r][out_node] = ad

    out.


    return new_sim, nr

def RS_getClusters(sim,alpha,level):
    #Data initialization
    similarity_hashmap = sim
    source_data = sim.keys()
    num_root = 0

    #Create nearest neighbor directory
    nn = get_nearest_neighbors(similarity_hashmap)

    ##
    # Search Roots
    roots=[]
 #   for i in level:
 #       for n in


if __name__ == '__main__':


