import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
df = pd.read_csv("placement.csv")

print("Dataset loaded successfully\n")
print(df.head())

# Step 2: Split input and output
X = df[['CGPA', 'Internships', 'Skills']]
y = df['Placed']

# Step 3: Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 4: Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Test model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Step 6: Take user input (REAL-TIME prediction)
print("\n========== Student Details Input ==========")

cgpa = float(input("Enter CGPA (example: 8.5): "))
internships = int(input("Enter internships (0 or 1 only): "))
skills = int(input("Enter skills (0 or 1 only): "))

print("==========================================")

# Step 7: Predict placement
result = model.predict([[cgpa, internships, skills]])

if result[0] == 1:
    print("\nResult: Student will be PLACED")
else:
    print("\nResult: Student will NOT be PLACED")