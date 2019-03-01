from AP.affinity_prop import *
from AP.utils import *

if __name__ == "__main__":

    # {"breast-w", "ecoli", "glass", "ionosphere", "iris", "kdd_synthetic_control", "mfeat-fourier", "mfeat-karhunen","mfeat-zernike"};
    # {"optdigits", "segment", "sonar", "vehicle", "waveform-5000", "letter", "kdd_synthetic_control"};
    # {pendigits, page-blocks,spambase, cmc}
    # {ERA,ESL, LEV, SWD}
    # {dermatology, diabetes, heart-statlog, liver-disorders, wine}
    # dataset = 'optdigits'
    # k = 3

    # file_results = '/Users/wenboxie/Data/rs-exp/ap/ap-' + dataset + '-ri.txt'

    file_results = '/Users/wenboxie/Data/rs-exp/ap/ap-Olivetti-ri.txt'

    write_results = open(file_results, 'w')
    # for k in range(2, 21):
    # file_data = open('/Users/wenboxie/Data/uci-20070111/exp/' + dataset + '(data).txt')
    # file_data = open('/Users/wenboxie/Data/ssim_cwssim.csv','r')
    # line = file_data.readline()
    #
    # m = len(line.split(',')) - 1
    # n = 1
    # for line in file_data.readlines():
    #     n += 1
    # file_data.close()
    # X = np.zeros((n,m))
    # file_data = open('/
    # Users/wenboxie/Data/uci-20070111/exp/' + dataset + '(data).txt')
    X = np.zeros((400, 400))
    file_data = open('/Users/wenboxie/Data/ssim_cwssim.csv', 'r')
    for line in file_data.readlines():
        att = line.split(',')
        X[int(att[0]) - 1, int(att[1]) - 1] = float(att[3])

    ground_truth = []
    # file_label = open('/Users/wenboxie/Data/uci-20070111/exp/' + dataset + '(label).txt')
    file_label = open('/Users/wenboxie/Data/Olivetti(label).txt')
    for line in file_label.readlines():
        ground_truth.append(line.strip('\n'))
    # S = -1 * euclidean_distance(X, squared=True)

    # print(S)
    median_value = np.median(X) * 10
    np.fill_diagonal(X, median_value)
    S = -1 * X

    af_prop = AffinityProp(similarity_matrix=S, alpha=0.5)
    exemplar_indices, exemplar_assignments = af_prop.solve()
    # print(exemplar_assignments)
    k = len(exemplar_indices)
    # print (exemplar_assignments)
    pred = exemplar_assignments
    # print (np.unique(exemplar_assignments, return_counts=True))

    # RESULTS
    # THE INDICES OF THE EXEMPLARS IS 4 21 AND 22
    # THERE ARE 19, 20, 21 ELEMENTS IN EACH CLUSTER INCLUDING THE EXEMPLAR

    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0
    for m in range(0, len(ground_truth)):
        for n in range(0, m):
            if ground_truth[m] == ground_truth[n]:
                if pred[n] == pred[m]:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred[n] == pred[m]:
                    FP += 1
                else:
                    TN += 1

    ri = (TP + TN) / (TP + TN + FP + FN)
    print(TP, TN, FP, FN)
    print('k:', k, 'RI:', ri)
    write_results.write(str(k) + '\t' + str(ri) + '\n')

    write_results.close()
