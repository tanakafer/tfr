import  numpy as np

def APatk(labels):

    ap =0
    # Element TPseen/i
    APatk = []
    GTP = np.count_nonzero(labels == 1)
    if GTP !=0:
        for i in np.arange(len(labels)):
            k=i+1
            # print("labels: {}".format(labels[:k]))
            if labels[i] == 0:
                TPSeen = 0
            else:
                TPSeen= np.count_nonzero(labels[:k]==1)
            # print("TPSeen: {}".format(TPSeen))
            # print("k: {}".format(k))
            APatk.append(TPSeen/(k))
        # print("GTP list: {}".format(labels))
        # print("AP@k: {}".format(np.array(APatk)))
        # print("GTP: {}".format(GTP))
        ap= np.sum(np.array(APatk))/GTP
    else:
        APatk = [0 for i in labels.tolist()]

    return ap, np.array(APatk)


def evaluate_emb(emb, labels):
    """Evaluate embeddings based on Recall@k."""
    d_mat = get_distance_matrix(emb)
    names = []
    accs = []
    for k in [1, 2, 4, 8, 16]:
        names.append('Recall@%d' % k)
        correct, cnt = 0.0, 0.0
        for i in range(emb.shape[0]):
            d_mat[i, i] = 1e10
            nns = np.argpartition(d_mat[i], k)[:k]
            if any(labels[i] == labels[nn] for nn in nns):
                correct += 1
            cnt += 1
        accs.append(correct/cnt)
    return names, accs
