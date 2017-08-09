from NG import Index

#LFR-1000; football
dataname = 'football'
method='rs-opt'
vInf = Index.Index()
print(vInf.variationInformaiton('/Users/wenboxie/Data/network/'+dataname+'/'+method+'-grandtruth.csv'))