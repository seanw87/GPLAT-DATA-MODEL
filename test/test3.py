with open("data/test.txt", 'rb') as f:
    index = 0
    while True:
        index += 1
        line = f.readline()
        print(line.decode('utf-8'))
        print("ftell: {}".format(f.tell()))

        print(index, len(line))
        if not line:
            break