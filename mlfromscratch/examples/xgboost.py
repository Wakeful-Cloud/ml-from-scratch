from __future__ import division, print_function
from sklearn import datasets
from mlfromscratch.utils import train_test_split
from mlfromscratch.supervised_learning import XGBoost
from dill import dumps, loads

def main():
    # Load data
    data = datasets.load_iris()
    x = data.data
    y = data.target

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, seed=2)

    # Group testing data by category
    categories = [[], [], []]
    for index, category in enumerate(y_test):
        categories[category].append(x_test[index])

    # # Train
    # clf = XGBoost()
    # clf.fit(x_train, y_train)

    # print(clf.predict(categories[0]), clf.predict(categories[1]), clf.predict(categories[2]))

    # with open("training.pckl", "wb") as file:
    #     file.write(dumps(clf.trees))

    clf = {}
    with open("training.pckl", "rb") as file:
        clf = loads(file.read())

    # Loop over each iris category
    for category in categories:
        # Loop over each iris
        for row in category:
            # Loop 100 times to make things take longer
            for i in range(0, 10000):
                # Predict the entire category
                clf.predict([row])

if __name__ == "__main__":
    main()
