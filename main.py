import pandas as pd 
 
df = pd.read_csv("titanic.csv") 
 
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True) 
 
df['Embarked'].fillna('S', inplace = True) 
 
age1 = df[df['Pclass'] == 1]['Age'].median() 
age2 = df[df['Pclass'] == 2]['Age'].median() 
age3 = df[df['Pclass'] == 3]['Age'].median() 
 
def set_age(row): 
    if pd.isnull(row['Age']): 
        if row['Pclass'] == 1: 
            return age1 
        elif row['Pclass'] == 2: 
            return age2 
        elif row['Pclass'] == 3: 
            return age3 
    return row['Age'] 
 
df['Age'] = df.apply(set_age, axis = 1) 
 
def set_sex(sex): 
    if sex == 'male': 
        return 
    elif sex == 'female': 
        return 0 
 
df['Sex'] = df['Sex'].apply(set_sex) 
 
df[list(pd.get_dummies(df['Embarked']).columns)] = pd.get_dummies(df['Embarked']) 
 
df.drop('Embarked', axis = 1, inplace = True) 
 
df.info()
