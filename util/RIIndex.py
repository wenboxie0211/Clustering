from NG import Index

#
#LFR-1000; football
dataname = 'football'
method='rs'

line=open('/Users/wenboxie/Data/network/'+dataname+'/result-rs-opt-m.csv')
line.readline()
i=0
label={}
pred={}
for l in line:
    las=l.split(',')
    label[i]=las[1]
    pred[i]=las[2]
    i+=1

TP = 0; TN = 0; FP = 0; FN = 0
for m in range(0,len(label)):
    for n in range(0,m):
        if label[m]==label[n]:
            if pred[n]==pred[m]:
                TP+=1
            else:
                FN+=1
        else:
            if pred[n]==pred[m]:
                FP+=1
            else:
                TN+=1


print ('RI:',((TP + TN) / (TP + TN +FP + FN)))
print ('AA:',(TP / (TP + FP ) + TN / (TN + FN)) / 2)


vInf = Index.Index()
print('InfMutual:',vInf.variationInformaiton('/Users/wenboxie/Data/network/'+dataname+'/result-rs-opt-m.csv'))
