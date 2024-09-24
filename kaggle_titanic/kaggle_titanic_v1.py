"""
Variable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex
Age	Age in years
sibsp	# of siblings / spouses aboard the Titanic
parch	# of parents / children aboard the Titanic
ticket	Ticket number
fare	Passenger fare
cabin	Cabin number
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower
age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)
parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.
"""

# * SETUP
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# * EDA

df_train.info()
df_train.describe()

# Total 891 passengers, 342 survived


class TitanicModel:
    def __init__(self, model):
        self.model = model

    def preprocess(self, df):
        # Handle missing values (Age, Cabin, Embarked) by dropping the rows
        df_handled_missing = df.dropna(subset=["Embarked", "Age"])

        # Encode categorical variables (Sex, Embarked)
        df_encoded = pd.get_dummies(df_handled_missing, columns=["Sex", "Embarked"])

        # Feature Selection
        X = df_encoded.drop(
            columns=["PassengerId", "Survived", "Name", "Ticket", "Cabin"]
        )
        y = df_encoded["Survived"]

        return X, y

    def train(self, df_train):
        X, y = self.preprocess(df_train)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_valid)
        accuracy = accuracy_score(y_valid, y_pred)
        print(f"Validation Accuracy: {accuracy}")


titanic = TitanicModel(model=LogisticRegression())
titanic.train(df_train)
