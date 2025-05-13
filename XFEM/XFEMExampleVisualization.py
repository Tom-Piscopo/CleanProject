import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    filename = "ExampleResults.csv"
    data = np.loadtxt(filename, delimiter=",")
    x = data[0]
    undamagedTopDisplacements = data[1]
    damagedTopDisplacements = data[2]

    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.plot(x, undamagedTopDisplacements, "b-", label="Without Crack")
    ax.plot(x, damagedTopDisplacements, "r--", label="With Crack")
    ax.set_xlabel("Length [m]")
    ax.set_ylabel("Deflection [m]")
    ax.legend(loc="best")
    figure.suptitle('Comparing Identical Beams With and Without Crack')
    plt.show()


