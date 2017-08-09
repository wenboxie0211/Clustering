from numpy import *
# L is matrix
def ECTD(L):
    n = L[0].size
    ep = mat(ones((n,n)))/n
    L_plus = (L - ep).I + ep
    sim = zeros((n,n))
    VG = 0
    for k in range(0,n):
        VG += L[k,k]
    for i in range(0,n):
        for j in range(0,i):
            if i==j:
                sim[i,j] = 0
            else:
                sim[i,j] = VG * (L_plus[i,i] + L_plus[j,j] - 2*L_plus[i,j])
                sim[j,i] = sim[i,j]
    return sim

# data3 = mat(random.rand(5000, 5000))
#     # print (data3)
#     # print("random 5000")
# re=ECTD(data3)
# print('complete!')
# print (ECTD(data3))
# s = mat([[1,-1,0,0,0,0,0],
#          [-1,4,-1,-1,-1,0,0],
#          [0,-1,2,0,0,-1,0],
#          [0,-1,0,2,-1,0,0],
#          [0,-1,0,-1,3,-1,0],
#          [0,0,-1,0,-1,3,-1],
#          [0,0,0,0,0,-1,1]])

s = mat([[1,-1,0,0,0,0,0],
         [-1,2,-1,0,0,0,0],
         [0,-1,1,0,0,0,0],
         [0,0,0,0,0,0,0],
         [0,0,0,0,1,-1,0],
         [0,0,0,0,-1,2,-1],
         [0,0,0,0,0,-1,1]])

print (ECTD(s))