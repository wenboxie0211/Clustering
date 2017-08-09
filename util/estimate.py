
def rand_index(label, pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for m in range(0, len(label)):
        for n in range(0, m):
            if label[m] == label[n]:
                if pred[n] == pred[m]:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred[n] == pred[m]:
                    FP += 1
                else:
                    TN += 1
    return (TP + TN) / (TP + TN +FP + FN)