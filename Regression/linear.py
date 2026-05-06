import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Load Data
try:
    df = pd.read_csv('data/Salary_dataset.csv')
except FileNotFoundError:
    print("Error: no data.")
    exit()

if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

X = df['YearsExperience'].values
y = df['Salary'].values
print(df.head())
print("\n")

# 2. Visualisation

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', alpha=0.7, label='Data Aktual')
plt.title('Sebaran Data: Years of Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show(block=False)
plt.pause(2)
plt.close()

# 3. Gradient Descent

def gradient_descent(X, y, learning_rate=0.01, epochs=300):
    m = 0.0
    b = 0.0
    n = len(X)
    cost_history = []
    
    for i in range(epochs):

        y_pred = m * X + b
        
        
        D_m = (-2/n) * sum(X * (y - y_pred))
        D_b = (-2/n) * sum(y - y_pred)
        
        
        m = m - learning_rate * D_m
        b = b - learning_rate * D_b
        
        
        cost = (1/n) * sum((y - y_pred)**2)
        cost_history.append(cost)
            
    return m, b, cost_history


best_m, best_b, cost_history = gradient_descent(X, y)


print(f"BEST m (Slope/Gradien) : {best_m:.4f}")
print(f"BEST b (Intercept)     : {best_b:.4f}")


# 4. Visualisation 

plt.figure(figsize=(10, 6))

plt.scatter(X, y, color='blue', alpha=0.7, label='Data Aktual')

y_pred_final = best_m * X + best_b


ss_res = sum((y - y_pred_final)**2)
ss_tot = sum((y - np.mean(y))**2)

r2 = 1 - (ss_res / ss_tot)

print(f"R-squared (R2)         : {r2:.4f}")
print("\n")
plt.plot(X, y_pred_final, color='red', linewidth=2, label='Garis Regresi (Best Fit Line)')

for i in range(len(X)):
    plt.plot([X[i], X[i]], [y[i], y_pred_final[i]], color='gray', linestyle='--', alpha=0.6)

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()
