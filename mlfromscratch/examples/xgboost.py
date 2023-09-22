from __future__ import division, print_function
from sklearn import datasets
from mlfromscratch.utils import train_test_split
from mlfromscratch.supervised_learning import XGBoost
from dill import dumps, loads
from monotonic import monotonic
from os.path import dirname, join

# Get the directory name of this script
__dirname__ = dirname(__file__)

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
    with open(join(__dirname__, "..", "..", "training.pckl"), "rb") as file:
        clf = loads(file.read())

    # Log the header
    print("# [DESCRIPTION] status: entry status (start: starting category, finished: finished category)")
    print("# [DESCRIPTION] time: system monotonic time")
    print("# [DESCRIPTION] category: dataset category")
    print("status,time,category")

    # Loop over each iris category
    for category_index, category in enumerate(categories):
        # Loop over each iris
        for iris in category:
            # Log the entry
            print("start,%.9f,%d" % (monotonic(), category_index))

            # Loop 10000 times to make things take longer
            for i in range(0, 10000):
                # Predict the individual iris
                clf.predict([iris])

            # Log the entry
            print("finished,%.9f,%d" % (monotonic(), category_index))

if __name__ == "__main__":
    main()
