import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
from submission_script import *
from dataset_script import dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Ova e primerok od podatochnoto mnozestvo, za treniranje/evaluacija koristete ja
# importiranata promenliva dataset
dataset_sample = [['1', '35', '12', '5', '1', '100', '0'],
                  ['1', '29', '7', '5', '1', '96', '1'],
                  ['1', '50', '8', '1', '3', '132', '0'],
                  ['1', '32', '11.75', '7', '3', '750', '0'],
                  ['1', '67', '9.25', '1', '1', '42', '0']]

if __name__ == '__main__':

    data = []
    for record in dataset:
        float_record = list(map(float, record))
        data.append(float_record)

    split_point = int(0.85 * len(data))
    train_set = data[:split_point]
    test_set = data[split_point:]

    train_x = [example[:-1] for example in train_set]
    train_y = [int(example[-1]) for example in train_set]
    test_x = [example[:-1] for example in test_set]
    test_y = [int(example[-1]) for example in test_set]

    classifier = GaussianNB()
    classifier.fit(train_x, train_y)

    correct_predictions = 0
    for idx in range(len(test_x)):
        pred = classifier.predict([test_x[idx]])[0]
        if pred == test_y[idx]:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_x)
    print(accuracy)

    input_line = input().strip().split()
    input_data = [float(val) for val in input_line]

    predicted_class = classifier.predict([input_data])
    predicted_probability = classifier.predisct_proba([input_data])

    print(predicted_class[0])
    print(predicted_probability)

    # Na kraj potrebno e da napravite submit na podatochnoto mnozestvo,
    # klasifikatorot i encoderot so povik na slednite funkcii

    # submit na trenirachkoto mnozestvo
    submit_train_data(train_x, train_y)

    # submit na testirachkoto mnozestvo
    submit_test_data(test_x, test_y)

    # submit na klasifikatorot
    submit_classifier(classifier)

    # povtoren import na kraj / ne ja otstranuvajte ovaa linija
