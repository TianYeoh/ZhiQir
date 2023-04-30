file = './0.txt'
for line in open(file, encoding='utf-8').readlines():
    left = line.strip().split('\t')[0]
    right = line.strip().split('\t')[1]
    print(left)
    print(right)