import matplotlib.pyplot as plt
import pandas as pd
import sys
import io
from matplotlib.ticker import MaxNLocator
import numpy as np


# USAGE
# python plot_convergence.py results.csv
# do the plot of the convergence results
def main():
    if len(sys.argv) > 1:
        # Read from file
        df = pd.read_csv(sys.argv[1])
    else:
        # Read from stdin
        input_data = sys.stdin.read()
        if not input_data:
            print("No input data provided. Usage: ./test | python plot.py")
            return
        df = pd.read_csv(io.StringIO(input_data))

    print("Data received:")
    print(df)

    plt.figure(figsize=(10, 6))
    plt.plot(df["N"], df["Iterations"], "o-", linewidth=3, markersize=8)
    plt.title("#iterations vs Problem Size", fontsize=16, fontweight="bold")
    plt.xlabel("Problem Size (N)", fontsize=14)
    plt.ylabel("Iterations to Converge", fontsize=14)
    plt.tick_params(axis="both", labelsize=16)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # Ideally, AMG iterations should remain constant as size increases
    plt.ylim(bottom=0)

    plt.savefig("convergence_plot.png")
    print("Plot saved to convergence_plot.png")

    # Plot time taken vs problem size
    plt.figure(figsize=(10, 6))
    plt.plot(df["N"], df["Time(s)"], "o-", linewidth=3, markersize=8)
    plt.title("Time vs Problem Size", fontsize=16, fontweight="bold")
    plt.xlabel("Problem Size (N)", fontsize=14)
    plt.ylabel("Time (s)", fontsize=14)
    plt.tick_params(axis="both", labelsize=16)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.ylim(bottom=0)
    plt.savefig("time_plot.png")
    print("Plot saved to time_plot.png")

    plt.show()


if __name__ == "__main__":
    main()
