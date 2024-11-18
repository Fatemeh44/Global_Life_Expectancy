import matplotlib.pyplot as plt

def plot_trends(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Year'], df['Life Expectancy'], marker='o')
    plt.xlabel('Year')
    plt.ylabel('Life Expectancy')
    plt.title('Future Life Expectancy Predictions')
    plt.grid(True)
    plt.show()