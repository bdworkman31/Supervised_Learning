import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
print(df)
df.dtypes  # Check for feature datatypes

df = df.drop(['Surname', 'RowNumber'], axis=1) #Drop 'Surname' and 'RowNumber' column, as these are irrelvant features to target prediction

geo_unique = df['Geography'].nunique() #Explore number of unique values in 'Geography' column for 
print(geo_unique)

df = pd.get_dummies(df, drop_first= True) #One Hot Encode (OHE), Avoid high correlation confusion and drop first column

print(df)

#Check need for normalization of values in columns:
print(min(df['EstimatedSalary']), max(df['EstimatedSalary']))   #Check difference between max and min values in each column
print(min(df['Balance']), max(df['Balance']))
df_null = df[df.isnull().any(axis=1)]  #Checking for null values
print(df_null)

df.dropna(subset=['Tenure'],inplace = True) #Drop null values in Tenure Column and Double check 
print(df[df['Tenure'].isna()]) #There are no null values here 

df_null2 = df[df.isnull().any(axis=1)] #Double check for null values
print(df_null2) # there are nonecnt_hascrcard_0 = df['HasCrCard'][df['HasCrCard'] == 0].count() #Check total of int64 0 values in the HasCrCard column
print(cnt_hascrcard_0)

cnt_hascrcard_1 = df['HasCrCard'][df['HasCrCard'] == 1].count() #Check total of int64 1 values in the HasCrCard column
print(cnt_hascrcard_1)#Check Number of Products Column
cnt_num_0 = df['NumOfProducts'][df['NumOfProducts'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_num_1 = df['NumOfProducts'][df['NumOfProducts'] == 1].count() #Check total of int64 1 values in the NumOfProducts column
cnt_num_2 = df['NumOfProducts'][df['NumOfProducts'] == 2].count() #Check total of int64 2 values in the NumOfProducts column
cnt_num_3 = df['NumOfProducts'][df['NumOfProducts'] == 3].count() #Check total of int64 3 values in the NumOfProducts column
print(f'Count 0: {cnt_num_0}, Count 1: {cnt_num_1}, Count 2: {cnt_num_2}, Count 3: {cnt_num_3}')


#Check Germany Geography for imbalance
cnt_geo_ger_0 = df['Geography_Germany'][df['Geography_Germany'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_geo_ger_1 = df['Geography_Germany'][df['Geography_Germany'] == 1].count() #Check total of int64 0 values in the NumOfProducts column
print(f'Germany 0: {cnt_geo_ger_0}, Germany 1: {cnt_geo_ger_1}')

#Check Spain Geography
cnt_geo_spain_0 = df['Geography_Spain'][df['Geography_Spain'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_geo_spain_1 = df['Geography_Spain'][df['Geography_Spain'] == 1].count() #Check total of int64 0 values in the NumOfProducts column
print(f'Spain 0: {cnt_geo_spain_0}, Spain 1: {cnt_geo_spain_1}')

#Check Gender 
cnt_male_0 = df['Gender_Male'][df['Gender_Male'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_male_1 = df['Gender_Male'][df['Gender_Male'] == 1].count() #Check total of int64 0 values in the NumOfProducts column
print(f'Male 0: {cnt_male_0}, Male 1: {cnt_male_1}')

#Check the target column ('Exited')
cnt_exited_0 = df['Exited'][df['Exited'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_exited_1 = df['Exited'][df['Exited'] == 1].count() #Check total of int64 0 values in the NumOfProducts column
print(f'Exited 0: {cnt_male_0}, Exited 1: {cnt_male_1}')#Set up features and split into training and validation datasets

target = df['Exited']
features = df.drop('Exited', axis=1)

features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size = 0.25, random_state = 50) #Split into validation and training set

#Feature Scaling (Normalization)

numeric = ['CreditScore', 'EstimatedSalary', 'Balance']

scaler = StandardScaler()
scaler.fit(features_train[numeric])

features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])

pd.options.mode.chained_assignment = None # Silence Warning; but does not seem to work

#Train the model

model = RandomForestClassifier(random_state = 50, n_estimators= 80, max_depth = 8)
model.fit(features_train, target_train) #Train RandomForestClassifier Model
predicted_valid = model.predict(features_valid)

print(f1_score(target_valid, predicted_valid))
#Upsample underrepresented values: 
def upsample(features, target, repeat):
    features_zeros = features[features['Geography_Germany'] == 0]
    features_ones = features[features['Geography_Germany'] == 1]
    target_zeros = target[features['Geography_Germany'] == 0]
    target_ones = target[features['Geography_Germany'] == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=50
    )

    return features_upsampled, target_upsampled


features_upsampled, target_upsampled = upsample(
    features_train, target_train, 3
)
features_drop_custid = features_upsampled.drop('CustomerId', axis=1) #Drop CustomerId column
features_valid = features_valid.drop('CustomerId', axis=1)

model = RandomForestClassifier(random_state = 50, n_estimators= 90, max_depth = 30)
model.fit(features_drop_custid, target_upsampled) #Train RandomForestClassifier Model
predicted_valid = model.predict(features_valid)

print(f1_score(target_valid, predicted_valid))#Downsample overrepresented values: 
def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_new = pd.concat(
        [features_zeros]
        + [features_ones.sample(frac=fraction, random_state=50)]
    )
    target_new = pd.concat(
        [target_zeros]
        + [target_ones.sample(frac=fraction, random_state=50)]
    )

    features_new, target_new = shuffle(
        features_new, target_new, random_state=50
    )

    return features_new, target_new


features_new, target_new = downsample(
    features_drop_custid, target_upsampled, 0.7
)


model = RandomForestClassifier(random_state = 50, n_estimators= 90, max_depth = 30)
model.fit(features_new, target_new) #Train RandomForestClassifier Model
predicted_valid = model.predict(features_valid)

print(f1_score(target_valid, predicted_valid))#Upsample underrepresented values in Spain: 

#First check the discrepancies again: 

#Check Germany Geography for imbalance
cnt_geo_ger_0 = features_drop_custid['Geography_Germany'][features_drop_custid['Geography_Germany'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_geo_ger_1 = features_drop_custid['Geography_Germany'][features_drop_custid['Geography_Germany'] == 1].count() #Check total of int64 0 values in the NumOfProducts column
print(f'Germany 0: {cnt_geo_ger_0}, Germany 1: {cnt_geo_ger_1}')

#Check Spain Geography
cnt_geo_spain_0 = features_drop_custid['Geography_Spain'][features_drop_custid['Geography_Spain'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_geo_spain_1 = features_drop_custid['Geography_Spain'][features_drop_custid['Geography_Spain'] == 1].count() #Check total of int64 0 values in the NumOfProducts column
print(f'Spain 0: {cnt_geo_spain_0}, Spain 1: {cnt_geo_spain_1}')



def upsample(features, target, repeat):
    features_zeros = features_drop_custid[features_drop_custid['Geography_Spain'] == 0]
    features_ones = features_drop_custid[features_drop_custid['Geography_Spain'] == 1]
    target_zeros = target[features_drop_custid['Geography_Spain'] == 0]
    target_ones = target[features_drop_custid['Geography_Spain'] == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=50
    )

    return features_upsampled, target_upsampled


features_up_spain, target_up_spain = upsample(
    features_drop_custid, target_upsampled, 3
)

model = RandomForestClassifier(random_state = 50, n_estimators= 90, max_depth = 30)
model.fit(features_up_spain, target_up_spain) #Train RandomForestClassifier Model
predicted_valid = model.predict(features_valid)

print(f1_score(target_valid, predicted_valid))



#Check Spain Geography
cnt_geo_spain_0 = features_up_spain['Geography_Spain'][features_up_spain['Geography_Spain'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_geo_spain_1 = features_up_spain['Geography_Spain'][features_up_spain['Geography_Spain'] == 1].count() #Check total of int64 0 values in the NumOfProducts column
print(f'Spain 0: {cnt_geo_spain_0}, Spain 1: {cnt_geo_spain_1}')

#Check Germany
cnt_geo_ger_0 = features_drop_custid['Geography_Germany'][features_drop_custid['Geography_Germany'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_geo_ger_1 = features_drop_custid['Geography_Germany'][features_drop_custid['Geography_Germany'] == 1].count() #Check total of int64 0 values in the NumOfProducts column
print(f'Germany 0: {cnt_geo_ger_0}, Germany 1: {cnt_geo_ger_1}')

#Check Number of Products Column
cnt_num_0 = df['NumOfProducts'][df['NumOfProducts'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_num_1 = df['NumOfProducts'][df['NumOfProducts'] == 1].count() #Check total of int64 1 values in the NumOfProducts column
cnt_num_2 = df['NumOfProducts'][df['NumOfProducts'] == 2].count() #Check total of int64 2 values in the NumOfProducts column
cnt_num_3 = df['NumOfProducts'][df['NumOfProducts'] == 3].count() #Check total of int64 3 values in the NumOfProducts column
print(f'Count 0: {cnt_num_0}, Count 1: {cnt_num_1}, Count 2: {cnt_num_2}, Count 3: {cnt_num_3}')#Upsample Number of Products column equal to 3

def upsample(features, target, repeat):
    features_ones = features_up_spain[features_up_spain['NumOfProducts'] == 1]
    features_twos = features_up_spain[features_up_spain['NumOfProducts'] == 2]
    features_threes = features_up_spain[features_up_spain['NumOfProducts'] == 3]
    target_ones = target[features_up_spain['NumOfProducts'] == 1]
    target_twos = target[features_up_spain['NumOfProducts'] == 2]
    target_threes = target[features_up_spain['NumOfProducts'] == 3]

    features_upsampled = pd.concat([features_ones] + [features_twos] + [features_threes] * repeat)
    target_upsampled = pd.concat([target_ones] + [target_twos] + [target_threes] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=50
    )

    return features_upsampled, target_upsampled


features_up_nums, target_up_nums = upsample(
    features_up_spain, target_up_spain, 2
)

model = RandomForestClassifier(random_state = 50, n_estimators= 89, max_depth = 19)
model.fit(features_up_nums, target_up_nums) #Train RandomForestClassifier Model
predicted_valid = model.predict(features_valid)

print(f1_score(target_valid, predicted_valid))#Downsample the 'NumOfProducts' column with class 2.  

def downsample(features, target, fraction):
    features_ones = features_up_nums[features_up_nums['NumOfProducts'] == 1]
    features_twos = features_up_nums[features_up_nums['NumOfProducts'] == 2]
    features_threes = features_up_nums[features_up_nums['NumOfProducts'] == 3]
    target_ones = target[features_up_nums['NumOfProducts'] == 1]
    target_twos = target[features_up_nums['NumOfProducts'] == 2]
    target_threes = target[features_up_nums['NumOfProducts'] == 3]

    features_downsampled = pd.concat(
        [features_ones] + [features_twos.sample(frac=fraction, random_state=50)]
        + [features_threes]
    )
    target_downsampled = pd.concat(
        [target_ones] + [target_twos.sample(frac=fraction, random_state=50)]
        + [target_threes]
    )

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=50
    )

    return features_downsampled, target_downsampled


features_downsampled, target_downsampled = downsample(
    features_up_nums, target_up_nums, 0.1
)

model = RandomForestClassifier(random_state = 50, n_estimators= 87, max_depth = 19)
model.fit(features_downsampled, target_downsampled) #Train RandoForestClassifier Model
predicted_valid = model.predict(features_valid)

print(f1_score(target_valid, predicted_valid))probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

for threshold in np.arange(0, 0.40, 0.0219):
    predicted_valid = probabilities_one_valid > threshold
    precision = precision_score(target_valid, predicted_valid)
    recall = recall_score(target_valid, predicted_valid)

print(f1_score(target_valid, predicted_valid))#Check Spain Geography
cnt_geo_spain_0 = features_downsampled['Geography_Spain'][features_downsampled['Geography_Spain'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_geo_spain_1 = features_downsampled['Geography_Spain'][features_downsampled['Geography_Spain'] == 1].count() #Check total of int64 0 values in the NumOfProducts column
print(f'Spain 0: {cnt_geo_spain_0}, Spain 1: {cnt_geo_spain_1}')

#Check Germany
cnt_geo_ger_0 = features_downsampled['Geography_Germany'][features_downsampled['Geography_Germany'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_geo_ger_1 = features_downsampled['Geography_Germany'][features_downsampled['Geography_Germany'] == 1].count() #Check total of int64 0 values in the NumOfProducts column
print(f'Germany 0: {cnt_geo_ger_0}, Germany 1: {cnt_geo_ger_1}')

#Check Number of Products Column
cnt_num_0 = features_downsampled['NumOfProducts'][features_downsampled['NumOfProducts'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_num_1 = features_downsampled['NumOfProducts'][features_downsampled['NumOfProducts'] == 1].count() #Check total of int64 1 values in the NumOfProducts column
cnt_num_2 = features_downsampled['NumOfProducts'][features_downsampled['NumOfProducts'] == 2].count() #Check total of int64 2 values in the NumOfProducts column
cnt_num_3 = features_downsampled['NumOfProducts'][features_downsampled['NumOfProducts'] == 3].count() #Check total of int64 3 values in the NumOfProducts column
print(f'Count 0: {cnt_num_0}, Count 1: {cnt_num_1}, Count 2: {cnt_num_2}, Count 3: {cnt_num_3}')

#Check Gender 
cnt_male_0 = features_downsampled['Gender_Male'][features_downsampled['Gender_Male'] == 0].count() #Check total of int64 0 values in the NumOfProducts column
cnt_male_1 = features_downsampled['Gender_Male'][features_downsampled['Gender_Male'] == 1].count() #Check total of int64 0 values in the NumOfProducts column
print(f'Male 0: {cnt_male_0}, Male 1: {cnt_male_1}')

#Check hascrcard
cnt_hascrcard_0 = features_downsampled['HasCrCard'][features_downsampled['HasCrCard'] == 0].count() #Check total of int64 0 values in the HasCrCard column
cnt_hascrcard_1 = features_downsampled['HasCrCard'][features_downsampled['HasCrCard'] == 1].count() #Check total of int64 1 values in the HasCrCard column
print(f'Has Card 0: {cnt_hascrcard_0}, Has Card 1: {cnt_hascrcard_1}')def downsample2(features, target, fraction):
    features_zeros = features[features['HasCrCard'] == 0]
    features_ones = features[features['HasCrCard'] == 1]
    target_zeros = target[features['HasCrCard'] == 0]
    target_ones = target[features['HasCrCard'] == 1]

    features_downsampled = pd.concat([features_zeros] + [features_ones.sample(frac=fraction, random_state=50)])
    target_downsampled = pd.concat([target_zeros] + [target_ones.sample(frac=fraction, random_state=50)])

    features_downsampled_cr, target_downsampled_cr = shuffle(
        features_downsampled, target_downsampled, random_state=50
    )

    return features_downsampled_cr, target_downsampled_cr

features_downsampled_cr, target_downsampled_cr = downsample2(
    features_downsampled, target_downsampled, 0.25
)

model = RandomForestClassifier(random_state = 50, n_estimators= 87, max_depth = 19)
model.fit(features_downsampled_cr, target_downsampled_cr) #Train RandomForestClassifier Model
predicted_valid = model.predict(features_valid)

print(f1_score(target_valid, predicted_valid))

#Check hascrcard
cnt_hascrcard_0 = features_downsampled_cr['HasCrCard'][features_downsampled_cr['HasCrCard'] == 0].count() #Check total of int64 0 values in the HasCrCard column
cnt_hascrcard_1 = features_downsampled_cr['HasCrCard'][features_downsampled_cr['HasCrCard'] == 1].count() #Check total of int64 1 values in the HasCrCard column
print(f'Has Card 0: {cnt_hascrcard_0}, Has Card 1: {cnt_hascrcard_1}')probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

for threshold in np.arange(0, 0.40, 0.0219):
    predicted_valid = probabilities_one_valid > threshold
    precision = precision_score(target_valid, predicted_valid)
    recall = recall_score(target_valid, predicted_valid)

print(f1_score(target_valid, predicted_valid))probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
f1_score = f1_score(target_valid, predicted_valid)

print(f'The AUC-ROC Score is {auc_roc} \nThe f1_score is {f1_score}')
print('\nPerformance Indicators: \n0.5-0.6: Poor \n0.6-0.7: Fair \n0.7-0.8: Good \n0.8-0.9: Excellent \nAbove 0.9: Outstanding')
