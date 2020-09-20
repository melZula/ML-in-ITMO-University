N, M, K = map(int, input().split())
C = enumerate(list(map(int, input().strip().split())))
C = sorted(C, key=lambda x: x[1])
res = list([] for x in range(K))
for i, v in enumerate(C):
    res[i % K].append(v[0] + 1)
for part in res:
    print(str(len(part)) + ' ' + ' '.join([str(p) for p in part]))