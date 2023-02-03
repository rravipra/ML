##############
# Name: Rthvik Raviprakash
# email: rthvik.07@gmail.com

import pandas as pd
import numpy as np

#C:\Users\user\Downloads\CS37300 HW2 (1)\CS37300 HW2\sample.data
#C:\\Users\\user\\Downloads\\HW3-CS373-V2\\HW3-CS373\\titanic-train.data

X = pd.read_csv('C:\\Users\\user\\Downloads\\HW3-CS373-V2\\HW3-CS373\\titanic-train.data', delimiter=',',
                    index_col=None, engine='python')
Y = pd.read_csv('C:\\Users\\user\\Downloads\\HW3-CS373-V2\\HW3-CS373\\titanic-train.label', delimiter=',',
                    index_col=None, engine='python')

X_test = pd.read_csv('C:\\Users\\user\\Downloads\\HW3-CS373-V2\\HW3-CS373\\titanic-test.data', delimiter=',',
                    index_col=None, engine='python')
Y_test = pd.read_csv('C:\\Users\\user\\Downloads\\HW3-CS373-V2\\HW3-CS373\\titanic-test.label', delimiter=',',
                    index_col=None, engine='python')
print(X)
for col in X_test.columns:
    X_test[col].fillna(X_test[col].mode()[0], inplace=True)

for column in X.columns:
    X[column].fillna(X[column].mode()[0], inplace=True)

data = pd.concat([X, Y], axis=1)
data_test = pd.concat([X_test,Y_test], axis=1)

def smooth(val, k):
    numerator = val.numerator
    denominator = val.denominator

    return (numerator + 1)/(denominator + k)

df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
df5 = pd.DataFrame()
df6 = pd.DataFrame()
df7 = pd.DataFrame()

def Pclass(data):
    df1 = pd.DataFrame()
    mean_df1 = data['Pclass'].mean()

    array = []

    array.append('Less')
    array.append('Less')
    array.append('Greater')
    array.append('Greater')

    label = []

    for k in range(2):
        label.append(0)
        label.append(1)

    counts = dict()

    counts['Less'] = [0, 0, 0]
    counts['Greater'] = [0, 0, 0]

    val = np.array(data['Pclass'].values)

    x = 0
    y = 0
    for t in range(len(val)):
        if val[t] <= mean_df1:
            x += 1
        else:
            y += 1
    prob_val = []
    prob_val.append(x / (x + y))
    prob_val.append(x / (x + y))
    prob_val.append(y / (x + y))
    prob_val.append(y / (x + y))

    for p in range(len(val)):
        if data._get_value(p, len(data.columns) - 1, takeable=True) == 0:
            if val[p] <= mean_df1:
                counts['Less'][0] += 1
                counts['Less'][1] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][1] += 1

        if data._get_value(p, len(data.columns) - 1, takeable=True) == 1:
            if val[p] <= mean_df1:
                counts['Less'][0] += 1
                counts['Less'][2] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][2] += 1

    prob = []
    prob_smooth = []

    for values in counts:
        if counts[values][0] == 0:
            prob.append(0)
            prob.append(0)
            prob_smooth.append(0.333)
            prob_smooth.append(0.333)
        else:
            prob.append((counts[values][1]) / (counts[values][0]))
            prob.append((counts[values][2]) / (counts[values][0]))
            prob_smooth.append((counts[values][1] + 1) / (counts[values][0] + 2))
            prob_smooth.append((counts[values][2] + 1) / (counts[values][0] + 2))

    df1 = pd.DataFrame(list(zip(label, array, prob, prob_smooth, prob_val)),
                       columns=['Label', 'Pclass', 'Probability', 'Prob_smooth', 'Prob_values'])
    return df1

def Sex(data):
    df2 = pd.DataFrame()
    mean_df2 = data['Sex'].mean()

    array = []

    array.append('Less')
    array.append('Less')
    array.append('Greater')
    array.append('Greater')

    label = []

    for k in range(2):
        label.append(0)
        label.append(1)

    counts = dict()

    counts['Less'] = [0, 0, 0]
    counts['Greater'] = [0, 0, 0]

    val = np.array(data['Sex'].values)

    x = 0
    y = 0
    for t in range(len(val)):
        if val[t] <= mean_df2:
            x += 1
        else:
            y += 1
    prob_val = []
    prob_val.append(x / (x + y))
    prob_val.append(x / (x + y))
    prob_val.append(y / (x + y))
    prob_val.append(y / (x + y))

    for p in range(len(val)):
        if data._get_value(p, len(data.columns) - 1, takeable=True) == 0:
            if val[p] <= mean_df2:
                counts['Less'][0] += 1
                counts['Less'][1] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][1] += 1

        if data._get_value(p, len(data.columns) - 1, takeable=True) == 1:
            if val[p] <= mean_df2:
                counts['Less'][0] += 1
                counts['Less'][2] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][2] += 1

    prob = []
    prob_smooth = []

    for values in counts:
        if counts[values][0] == 0:
            prob.append(0)
            prob.append(0)
            prob_smooth.append(0.333)
            prob_smooth.append(0.333)
        else:
            prob.append((counts[values][1]) / (counts[values][0]))
            prob.append((counts[values][2]) / (counts[values][0]))
            prob_smooth.append((counts[values][1] + 1) / (counts[values][0] + 2))
            prob_smooth.append((counts[values][2] + 1) / (counts[values][0] + 2))

    df2 = pd.DataFrame(list(zip(label, array, prob, prob_smooth, prob_val)),
                       columns=['Label', 'Sex', 'Probability', 'Prob_smooth', 'Prob_values'])
    return df2

def Age(data):
    df3 = pd.DataFrame()
    mean_df3 = data['Age'].mean()

    array = []

    array.append('Less')
    array.append('Less')
    array.append('Greater')
    array.append('Greater')

    label = []

    for k in range(2):
        label.append(0)
        label.append(1)

    counts = dict()

    counts['Less'] = [0, 0, 0]
    counts['Greater'] = [0, 0, 0]

    val = np.array(data['Age'].values)

    x = 0
    y = 0
    for t in range(len(val)):
        if val[t] <= mean_df3:
            x += 1
        else:
            y += 1
    prob_val = []
    prob_val.append(x / (x + y))
    prob_val.append(x / (x + y))
    prob_val.append(y / (x + y))
    prob_val.append(y / (x + y))

    for p in range(len(val)):
        if data._get_value(p, len(data.columns) - 1, takeable=True) == 0:
            if val[p] <= mean_df3:
                counts['Less'][0] += 1
                counts['Less'][1] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][1] += 1

        if data._get_value(p, len(data.columns) - 1, takeable=True) == 1:
            if val[p] <= mean_df3:
                counts['Less'][0] += 1
                counts['Less'][2] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][2] += 1

    prob = []
    prob_smooth = []

    for values in counts:
        if counts[values][0] == 0:
            prob.append(0)
            prob.append(0)
            prob_smooth.append(0.333)
            prob_smooth.append(0.333)
        else:
            prob.append((counts[values][1]) / (counts[values][0]))
            prob.append((counts[values][2]) / (counts[values][0]))
            prob_smooth.append((counts[values][1] + 1) / (counts[values][0] + 2))
            prob_smooth.append((counts[values][2] + 1) / (counts[values][0] + 2))

    df3 = pd.DataFrame(list(zip(label, array, prob, prob_smooth, prob_val)),
                       columns=['Label', 'Age', 'Probability', 'Prob_smooth', 'Prob_values'])
    return df3

def Fare(data):
    df4 = pd.DataFrame()
    mean_df4 = data['Fare'].mean()

    array = []

    array.append('Less')
    array.append('Less')
    array.append('Greater')
    array.append('Greater')

    label = []

    for k in range(2):
        label.append(0)
        label.append(1)

    counts = dict()

    counts['Less'] = [0, 0, 0]
    counts['Greater'] = [0, 0, 0]

    val = np.array(data['Fare'].values)

    x = 0
    y = 0
    for t in range(len(val)):
        if val[t] <= mean_df4:
            x += 1
        else:
            y += 1
    prob_val = []
    prob_val.append(x / (x + y))
    prob_val.append(x / (x + y))
    prob_val.append(y / (x + y))
    prob_val.append(y / (x + y))

    for p in range(len(val)):
        if data._get_value(p, len(data.columns) - 1, takeable=True) == 0:
            if val[p] <= mean_df4:
                counts['Less'][0] += 1
                counts['Less'][1] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][1] += 1

        if data._get_value(p, len(data.columns) - 1, takeable=True) == 1:
            if val[p] <= mean_df4:
                counts['Less'][0] += 1
                counts['Less'][2] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][2] += 1

    prob = []
    prob_smooth = []

    for values in counts:
        if counts[values][0] == 0:
            prob.append(0)
            prob.append(0)
            prob_smooth.append(0.333)
            prob_smooth.append(0.333)
        else:
            prob.append((counts[values][1]) / (counts[values][0]))
            prob.append((counts[values][2]) / (counts[values][0]))
            prob_smooth.append((counts[values][1] + 1) / (counts[values][0] + 2))
            prob_smooth.append((counts[values][2] + 1) / (counts[values][0] + 2))

    df4 = pd.DataFrame(list(zip(label, array, prob, prob_smooth, prob_val)),
                       columns=['Label', 'Fare', 'Probability', 'Prob_smooth', 'Prob_values'])
    return df4

def Embarked(data):
    df5 = pd.DataFrame()
    mean_df5 = data['Embarked'].mean()

    array = []

    array.append('Less')
    array.append('Less')
    array.append('Greater')
    array.append('Greater')

    label = []

    for k in range(2):
        label.append(0)
        label.append(1)

    counts = dict()

    counts['Less'] = [0, 0, 0]
    counts['Greater'] = [0, 0, 0]

    val = np.array(data['Embarked'].values)

    x = 0
    y = 0
    for t in range(len(val)):
        if val[t] <= mean_df5:
            x += 1
        else:
            y += 1
    prob_val = []
    prob_val.append(x / (x + y))
    prob_val.append(x / (x + y))
    prob_val.append(y / (x + y))
    prob_val.append(y / (x + y))

    for p in range(len(val)):
        if data._get_value(p, len(data.columns) - 1, takeable=True) == 0:
            if val[p] <= mean_df5:
                counts['Less'][0] += 1
                counts['Less'][1] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][1] += 1

        if data._get_value(p, len(data.columns) - 1, takeable=True) == 1:
            if val[p] <= mean_df5:
                counts['Less'][0] += 1
                counts['Less'][2] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][2] += 1

    prob = []
    prob_smooth = []

    for values in counts:
        if counts[values][0] == 0:
            prob.append(0)
            prob.append(0)
            prob_smooth.append(0.333)
            prob_smooth.append(0.333)
        else:
            prob.append((counts[values][1]) / (counts[values][0]))
            prob.append((counts[values][2]) / (counts[values][0]))
            prob_smooth.append((counts[values][1] + 1) / (counts[values][0] + 2))
            prob_smooth.append((counts[values][2] + 1) / (counts[values][0] + 2))

    df5 = pd.DataFrame(list(zip(label, array, prob, prob_smooth, prob_val)),
                       columns=['Label', 'Embarked', 'Probability', 'Prob_smooth', 'Prob_values'])
    return df5

def relatives(data):
    df6 = pd.DataFrame()
    mean_df6 = data['relatives'].mean()

    array = []

    array.append('Less')
    array.append('Less')
    array.append('Greater')
    array.append('Greater')

    label = []

    for k in range(2):
        label.append(0)
        label.append(1)

    counts = dict()

    counts['Less'] = [0, 0, 0]
    counts['Greater'] = [0, 0, 0]

    val = np.array(data['relatives'].values)

    x = 0
    y = 0
    for t in range(len(val)):
        if val[t] <= mean_df6:
            x += 1
        else:
            y += 1
    prob_val = []
    prob_val.append(x/(x+y))
    prob_val.append(x / (x + y))
    prob_val.append(y / (x + y))
    prob_val.append(y / (x + y))

    for p in range(len(val)):
        if data._get_value(p, len(data.columns) - 1, takeable=True) == 0:
            if val[p] <= mean_df6:
                counts['Less'][0] += 1
                counts['Less'][1] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][1] += 1

        if data._get_value(p, len(data.columns) - 1, takeable=True) == 1:
            if val[p] <= mean_df6:
                counts['Less'][0] += 1
                counts['Less'][2] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][2] += 1

    prob = []
    prob_smooth = []

    for values in counts:
        if counts[values][0] == 0:
            prob.append(0)
            prob.append(0)
            prob_smooth.append(0.333)
            prob_smooth.append(0.333)
        else:
            prob.append((counts[values][1]) / (counts[values][0]))
            prob.append((counts[values][2]) / (counts[values][0]))
            prob_smooth.append((counts[values][1] + 1) / (counts[values][0] + 2))
            prob_smooth.append((counts[values][2] + 1) / (counts[values][0] + 2))

    df6 = pd.DataFrame(list(zip(label, array, prob, prob_smooth, prob_val)),
                       columns=['Label', 'relatives', 'Probability', 'Prob_smooth', 'Prob_values'])
    return df6

def IsAlone(data):
    df7 = pd.DataFrame()
    mean_df7 = data['IsAlone'].mean()

    array = []

    array.append('Less')
    array.append('Less')
    array.append('Greater')
    array.append('Greater')

    label = []

    for k in range(2):
        label.append(0)
        label.append(1)

    counts = dict()

    counts['Less'] = [0, 0, 0]
    counts['Greater'] = [0, 0, 0]

    val = np.array(data['IsAlone'].values)

    x = 0
    y = 0
    for t in range(len(val)):
        if val[t] <= mean_df7:
            x += 1
        else:
            y += 1
    prob_val = []
    prob_val.append(x / (x + y))
    prob_val.append(x / (x + y))
    prob_val.append(y / (x + y))
    prob_val.append(y / (x + y))

    for p in range(len(val)):
        if data._get_value(p, len(data.columns) - 1, takeable=True) == 0:
            if val[p] <= mean_df7:
                counts['Less'][0] += 1
                counts['Less'][1] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][1] += 1

        if data._get_value(p, len(data.columns) - 1, takeable=True) == 1:
            if val[p] <= mean_df7:
                counts['Less'][0] += 1
                counts['Less'][2] += 1
            else:
                counts['Greater'][0] += 1
                counts['Greater'][2] += 1

    prob = []
    prob_smooth = []

    for values in counts:
        if counts[values][0] == 0:
            prob.append(0)
            prob.append(0)
            prob_smooth.append(0.333)
            prob_smooth.append(0.333)
        else:
            prob.append((counts[values][1]) / (counts[values][0]))
            prob.append((counts[values][2]) / (counts[values][0]))
            prob_smooth.append((counts[values][1] + 1) / (counts[values][0] + 2))
            prob_smooth.append((counts[values][2] + 1) / (counts[values][0] + 2))

    df7 = pd.DataFrame(list(zip(label, array, prob, prob_smooth, prob_val)),
                       columns=['Label', 'IsAlone', 'Probability', 'Prob_smooth', 'Prob_values'])
    return df7

def prob_label_zero(data):
    survived = data['survived']
    arr = survived.values
    l = len(arr)
    count = 0

    for val in arr:
        if val == 0:
            count += 1

    return count/l

print(Pclass(data))
print(Sex(data))
print(Age(data))
print(Fare(data))
print(Embarked(data))
print(relatives(data))
print(IsAlone(data))
#print(counts)
print(data)

def find_prob_arr(data_train, data_test):
    labels = []
    prob_arr = []

    for index in data_test.index:
        prob_0 = 1
        prob_1 = 1

        # Get Probability of Pclass
        df_1 = Pclass(data_train)
        mean_train_Pclass = data_train['Pclass'].mean()
        value_1 = data_test['Pclass'][index]
        if value_1 <= mean_train_Pclass:
            row_1_zero = df_1[(df_1['Label'] == 0) & (df_1['Pclass'] == 'Less')]
            row_1_one = df_1[(df_1['Label'] == 1) & (df_1['Pclass'] == 'Less')]
        else:
            row_1_zero = df_1[(df_1['Label'] == 0) & (df_1['Pclass'] == 'Greater')]
            row_1_one = df_1[(df_1['Label'] == 1) & (df_1['Pclass'] == 'Greater')]
        #print(row_1_zero)
        #print(row_1_one)
        val_1_zero = row_1_zero.iloc[0]['Prob_smooth']
        val_1_one = row_1_one.iloc[0]['Prob_smooth']
        val_1_z = row_1_zero.iloc[0]['Prob_values']
        val_1_o = row_1_one.iloc[0]['Prob_values']
        prob_0 = (prob_0 * val_1_zero)/ val_1_z
        prob_1 = (prob_1 * val_1_one)/ val_1_o

        # Get Probability of Sex
        df_2 = Sex(data_train)
        mean_train_Sex = data_train['Sex'].mean()
        value_2 = data_test['Sex'][index]
        if value_2 <= mean_train_Sex:
            row_2_zero = df_2[(df_2['Label'] == 0) & (df_2['Sex'] == 'Less')]
            row_2_one = df_2[(df_2['Label'] == 1) & (df_2['Sex'] == 'Less')]
        else:
            row_2_zero = df_2[(df_2['Label'] == 0) & (df_2['Sex'] == 'Greater')]
            row_2_one = df_2[(df_2['Label'] == 1) & (df_2['Sex'] == 'Greater')]
        val_2_zero = row_2_zero.iloc[0]['Prob_smooth']
        val_2_one = row_2_one.iloc[0]['Prob_smooth']
        val_2_z = row_2_zero.iloc[0]['Prob_values']
        val_2_o = row_2_one.iloc[0]['Prob_values']
        prob_0 = (prob_0 * val_2_zero)/ val_2_z
        prob_1 = (prob_1 * val_2_one)/val_2_o

        """val_2_prob = row_2.iloc[0]['Prob_values']
        prob = prob / val_2_prob"""

        # Get Probability of Age
        df_3 = Age(data_train)
        mean_train_Age = data_train['Age'].mean()
        value_3 = data_test['Age'][index]
        if value_3 <= mean_train_Age:
            row_3_zero = df_3[(df_3['Label'] == 0) & (df_3['Age'] == 'Less')]
            row_3_one = df_3[(df_3['Label'] == 1) & (df_3['Age'] == 'Less')]
        else:
            row_3_zero = df_3[(df_3['Label'] == 0) & (df_3['Age'] == 'Greater')]
            row_3_one = df_3[(df_3['Label'] == 1) & (df_3['Age'] == 'Greater')]
        val_3_zero = row_3_zero.iloc[0]['Prob_smooth']
        val_3_one = row_3_one.iloc[0]['Prob_smooth']
        val_3_z = row_3_zero.iloc[0]['Prob_values']
        val_3_o = row_3_one.iloc[0]['Prob_values']
        prob_0 = (prob_0 * val_3_zero) / val_3_z
        prob_1 = (prob_1 * val_3_one) / val_3_o

        # Get Probability of Fare
        df_4 = Fare(data_train)
        mean_train_Fare = data_train['Fare'].mean()
        value_4 = data_test['Fare'][index]
        if value_4 <= mean_train_Fare:
            row_4_zero = df_4[(df_4['Label'] == 0) & (df_4['Fare'] == 'Less')]
            row_4_one = df_4[(df_4['Label'] == 1) & (df_4['Fare'] == 'Less')]
        else:
            row_4_zero = df_4[(df_4['Label'] == 0) & (df_4['Fare'] == 'Greater')]
            row_4_one = df_4[(df_4['Label'] == 1) & (df_4['Fare'] == 'Greater')]
        val_4_zero = row_4_zero.iloc[0]['Prob_smooth']
        val_4_one = row_4_one.iloc[0]['Prob_smooth']
        val_4_z = row_4_zero.iloc[0]['Prob_values']
        val_4_o = row_4_one.iloc[0]['Prob_values']
        prob_0 = (prob_0 * val_4_zero) / val_4_z
        prob_1 = (prob_1 * val_4_one) / val_4_o

        # Get Probability of Embarked
        df_5 = Embarked(data_train)
        mean_train_Embarked = data_train['Embarked'].mean()
        value_5 = data_test['Embarked'][index]
        if value_5 <= mean_train_Embarked:
            row_5_zero = df_5[(df_5['Label'] == 0) & (df_5['Embarked'] == 'Less')]
            row_5_one = df_5[(df_5['Label'] == 1) & (df_5['Embarked'] == 'Less')]
        else:
            row_5_zero = df_5[(df_5['Label'] == 0) & (df_5['Embarked'] == 'Greater')]
            row_5_one = df_5[(df_5['Label'] == 1) & (df_5['Embarked'] == 'Greater')]
        val_5_zero = row_5_zero.iloc[0]['Prob_smooth']
        val_5_one = row_5_one.iloc[0]['Prob_smooth']
        val_5_z = row_5_zero.iloc[0]['Prob_values']
        val_5_o = row_5_one.iloc[0]['Prob_values']
        prob_0 = (prob_0 * val_5_zero) / val_5_z
        prob_1 = (prob_1 * val_5_one) / val_5_o
        """val_5_prob = row_5.iloc[0]['Prob_values']
        prob = prob / val_5_prob"""

        # Get Probability of relatives
        df_6 = relatives(data_train)
        mean_train_relatives = data_train['relatives'].mean()
        value_6 = data_test['relatives'][index]
        if value_6 <= mean_train_relatives:
            row_6_zero = df_6[(df_6['Label'] == 0) & (df_6['relatives'] == 'Less')]
            row_6_one = df_6[(df_6['Label'] == 1) & (df_6['relatives'] == 'Less')]
        else:
            row_6_zero = df_6[(df_6['Label'] == 0) & (df_6['relatives'] == 'Greater')]
            row_6_one = df_6[(df_6['Label'] == 1) & (df_6['relatives'] == 'Greater')]

        val_6_zero = row_6_zero.iloc[0]['Prob_smooth']
        val_6_one = row_6_one.iloc[0]['Prob_smooth']
        val_6_z = row_6_zero.iloc[0]['Prob_values']
        val_6_o = row_6_one.iloc[0]['Prob_values']
        prob_0 = (prob_0 * val_6_zero) / val_6_z
        prob_1 = (prob_1 * val_6_one) / val_6_o
        """val_6_prob = row_6.iloc[0]['Prob_values']
        prob = prob / val_6_prob"""

        # Get Probability of IsAlone
        df_7 = IsAlone(data_train)
        mean_train_IsAlone = data_train['IsAlone'].mean()
        value_7 = data_test['IsAlone'][index]
        if value_7 <= mean_train_IsAlone:
            row_7_zero = df_7[(df_7['Label'] == 0) & (df_7['IsAlone'] == 'Less')]
            row_7_one = df_7[(df_7['Label'] == 1) & (df_7['IsAlone'] == 'Less')]
        else:
            row_7_zero = df_7[(df_7['Label'] == 0) & (df_7['IsAlone'] == 'Greater')]
            row_7_one = df_7[(df_7['Label'] == 1) & (df_7['IsAlone'] == 'Greater')]
        val_7_zero = row_7_zero.iloc[0]['Prob_smooth']
        val_7_one = row_7_one.iloc[0]['Prob_smooth']
        val_7_z = row_7_zero.iloc[0]['Prob_values']
        val_7_o = row_7_one.iloc[0]['Prob_values']
        prob_0 = (prob_0 * val_7_zero) / val_7_z
        prob_1 = (prob_1 * val_7_one) / val_7_o
        """val_7_prob = row_7.iloc[0]['Prob_values']
        prob = prob / val_7_prob"""

        prob_0 = prob_0 * prob_label_zero(data)
        prob_1 = prob_1 * (1 - prob_label_zero(data))

        """if data[index]['survived'] == 0:
            prob_arr.append(prob_0 / (prob_0 + prob_1))

        if data[index]['survived'] == 1:
            prob_arr.append(prob_1 / (prob_0 + prob_1))"""

        if data_test['survived'][index] == 0:
            if prob_0 >= 0.5:
                labels.append(0)
            else:
                labels.append(1)

            prob_arr.append(prob_0)

        else:
            if prob_1 >= 0.5:
                labels.append(1)
            else:
                labels.append(0)

            prob_arr.append(prob_1)


        #print(index)

    return prob_arr, labels

def zero_one_loss(arr,data):
    labels = np.array(data['survived'].values)
    k = 0
    l = len(arr)

    for i in range(l):
        if not arr[i] == labels[i]:
            k += 1

    print("ZERO-ONE LOSS=", k/l)
    #return k/l

def squared_loss(arr):
    l = len(arr)

    sum_squared = 0
    for val in arr:
        sum_squared = sum_squared + ((1-val)*(1-val))

    print("SQUARED LOSS=", sum_squared/l)
    #return sum_squared/l

def accuracy(arr1, arr2):

    k = 0

    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            k += 1

    print("Test Accuracy=", k/ len(arr1))

#print(find_prob_arr(data,data_test)[0])
print('acc= ', accuracy(find_prob_arr(data,data_test)[1], data_test['survived'].values))
#print(accuracy(find_prob_arr(data)))
#print(find_prob_arr(data))
#print(find_prob_arr(data, data_test)[1])
#print(data['survived'].values)

#print('Accuracy = ', accuracy(data_test['survived'].values, find_prob_arr(data, data_test)[1]))
#print('Zero-one Loss = ', zero_one_loss(find_prob_arr(data, data_test)[1], data_test))

#print(accuracy(find_prob_arr(data)[0]))
#print(find_prob_arr(data, data_test)[0])

#print('Squared Loss = ', squared_loss(find_prob_arr(data, data_test)[0]))
#print(zero_one_loss(find_prob_arr(data)[1]))

def cross_validation(data, perc):
    sum = 0
    sum1 = 0
    for i in range(10):
        train = data.sample(frac = perc)
        test = data.drop(train.index)
        arr1 = find_prob_arr(train, test)[1]
        arr2 = find_prob_arr(train, test)[0]
        sum = sum + zero_one_loss(arr1, test)
        sum1 = sum1 + squared_loss(arr2)

    return sum/10, sum1/10

"""for i in [0.01, 0.10, 0.50]:
    print("Mean-Zero One Loss for " + str(i) + " : ", cross_validation(data, i)[0])
    print("Mean Squared Loss for " + str(i) + " : ", cross_validation(data, i)[1])"""

def count():
    val = Y['survived'].values
    zero = 0
    one = 0
    for i in range(len(val)):
        if val[i] == 0:
            zero += 1
        else:
            one+= 1
    return zero, one, len(val)

#print(cross_validation(data, 0.01))
print(count())
zero_one_loss(find_prob_arr(data, data_test)[1], data_test)
squared_loss(find_prob_arr(data, data_test)[0])
accuracy(data_test['survived'].values, find_prob_arr(data, data_test)[1])

if __name__ == "__main__":
    # parse arguments
    import argparse
    import sys

    """parser = argparse.ArgumentParser(description='CS373 Homework3 NBC')
    parser.add_argument('--trainFileData')
    parser.add_argument('--trainFileLabel')
    parser.add_argument('--testFileData')
    parser.add_argument('--testFileLabel')
    args = parser.parse_args()
    print(args)"""

    X_train = pd.read_csv(sys.argv[1], delimiter=',',index_col=None, engine='python')
    Y_train = pd.read_csv(sys.argv[2], delimiter=',',index_col=None, engine='python')

    X_test = pd.read_csv(sys.argv[3], delimiter=',', index_col=None, engine='python')
    Y_test = pd.read_csv(sys.argv[4], delimiter=',', index_col=None, engine='python')

    for column in X_train.columns:
        X_train[column].fillna(X_train[column].mode()[0], inplace=True)
    for col in X_test.columns:
        X_test[col].fillna(X_test[col].mode()[0], inplace=True)

    data = pd.concat([X_train, Y_train], axis=1)
    data_test = pd.concat([X_test, Y_test], axis=1)

    zero_one_loss(find_prob_arr(data, data_test)[1], data_test)
    squared_loss(find_prob_arr(data, data_test)[0])
    accuracy(data_test['survived'].values, find_prob_arr(data, data_test)[1])


