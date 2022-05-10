lines = open('20220509_233047.log', 'r').readlines()

max_accuracy = 0
for line in lines:
    line = line.rstrip()
    if 'accuracy_top-1' in line:
        accuracy = float(line.split('accuracy_top-1')[-1][2:6])

        if accuracy > max_accuracy:
            max_accuracy = accuracy

print(max_accuracy)
