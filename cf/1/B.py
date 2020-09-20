TP = []
FP = []
FN = []
F_micro = 0
F_macro = 0

def Prec(c):
    _gr = TP[c] + FP[c]
    return TP[c] / _gr if _gr != 0 else 0

def Recall(c):
    _gr = TP[c] + FN[c]
    return TP[c] / _gr if _gr != 0 else 0

def F_score(c, b=1):
    _prec = Prec(c)
    _recall = Recall(c)
    if _prec != 0 and _recall != 0:
        return (1 + pow(b, 2)) * _prec * _recall / (pow(b, 2) * _prec + _recall)
    else:
        return 0

K = int(input())
CM = list(list())
for x in range(K):
    CM.append(list(map(int, input().strip().split())))

All = sum(list(sum(e) for e in zip(*CM)))

for i in range(K):
    tp = CM[i][i]
    TP.append(tp)
    FN.append(sum(CM[i]) - tp)
    FP.append(sum(list(e[i] for e in CM)) - tp)
    F_micro += (sum(CM[i]) * F_score(i)) / All

def PrecW():
    _sum = 0
    for i in range(K):
        P = sum(list(e[i] for e in CM))
        _sum += CM[i][i] * sum(CM[i]) / P if P != 0 else 0
    return _sum / All

def RecallW():
    return sum(CM[i][i] for i in range(K)) / All

F_macro = 2 * PrecW() * RecallW() / (PrecW() + RecallW())
    
print(F_macro)
print(F_micro)
