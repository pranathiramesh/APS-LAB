from data import generate_linear_data
from visual import plot_2d_data

def main():
    x, y = generate_linear_data()
    plot_2d_data(x, y, title="Linearly Separable Data")

if __name__ == "__main__":
    main()