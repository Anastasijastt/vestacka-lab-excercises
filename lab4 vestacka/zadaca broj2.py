import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
from submission_script import *
from dataset_script import dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics

# Ova e primerok od podatochnoto mnozestvo, za treniranje/evaluacija koristete ja
# importiranata promenliva dataset
dataset_sample = [[180.0, 23.6, 25.2, 27.9, 25.4, 14.0, 'Roach'],
                  [12.2, 11.5, 12.2, 13.4, 15.6, 10.4, 'Smelt'],
                  [135.0, 20.0, 22.0, 23.5, 25.0, 15.0, 'Perch'],
                  [1600.0, 56.0, 60.0, 64.0, 15.0, 9.6, 'Pike'],
                  [120.0, 20.0, 22.0, 23.5, 26.0, 14.5, 'Perch']]

if __name__ == '__main__':
    # Vashiot kod tuka
    col_index = int(input())
    numbers_tree = int(input())
    criterion = input()

    train_set = dataset[:int(0.85 * len(dataset))]
    train_X = [row[:-1] for row in train_set]
    train_Y = [row[-1] for row in train_set]

    test_set = dataset[int(0.85 * len(dataset)):]
    test_X = [row[:-1] for row in test_set]
    test_Y = [row[-1] for row in test_set]

    classifier = RandomForestClassifier(n_estimators=numbers_tree, criterion=criterion, random_state=0)

    train_X_new = list()
    test_X_new = list()
    for t in train_X:
        row = [t[i] for i in range(len(t)) if i != col_index]
        train_X_new.append(row)

    for t in test_X:
        row = [t[i] for i in range(len(t)) if i != col_index]
        test_X_new.append(row)

    classifier.fit(train_X_new, train_Y)
    pred_Y = classifier.predict(test_X_new)
    accuracy = metrics.accuracy_score(test_X, pred_Y)
    print(f'Accuracy: {accuracy}')

    new_record = [el for el in input().split(' ')]
    new_record = [new_record[i] for i in range(len(new_record)) if i != col_index]

    prediction = classifier.predict([new_record])
    pred_features = classifier.predict_proba([new_record])
    print(prediction[0])
    print(pred_features[0])
    # Na kraj potrebno e da napravite submit na podatochnoto mnozestvo
    # i klasifikatorot so povik na slednite funkcii

    # submit na trenirachkoto mnozestvo
    submit_train_data(train_X_new, train_Y)

    # submit na testirachkoto mnozestvo
    submit_test_data(test_X_new, test_Y)

    # submit na klasifikatorot
    submit_classifier(classifier)
