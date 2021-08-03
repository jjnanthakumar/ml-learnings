import pandas as pd
import matplotlib.pyplot as plt

def gradient_descent(x,y):
    m_cur = b_cur = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.08
    plt.scatter(x,y,color='red',marker='+',linewidth=5)
    for i in range(iterations):
        y_pred = m_cur*x + b_cur
        plt.plot(x,y_pred,color='green')
        cost = (1/n)*sum(map(lambda x:x**2,y-y_pred))
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum(y-y_pred)
        m_cur = m_cur - learning_rate *md
        b_cur = b_cur - learning_rate *bd
        print(f'm {m_cur}, b {b_cur}, iteration {i}, cost {cost}')
    plt.show()


df = pd.read_csv('Data/test_scores.csv')
print(df.head())