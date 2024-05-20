import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score, KFold, train_test_split


def handle_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = dataframe[(dataframe[column] < lower) | (dataframe[column] > upper)]
    dataframe.loc[(dataframe[column] > upper), column] = upper
    dataframe.loc[(dataframe[column] < lower), column] = lower
    return dataframe

df_train = pd.read_csv('./datasets/train.csv')

# Handling Missing Values
features_with_na = [feature for feature in df_train.columns if df_train[feature].isnull().sum() > 0]
missing_data = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_columns = missing_data[missing_data > 0]
missing_data = pd.concat([missing_data, percent], axis=1, keys=['Total', 'Percent'])

# Numerical Values
numerical_features = [feature for feature in df_train.columns if df_train[feature].dtypes != 'O']
# List of variables that contain year information
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
# Identify discrete variables
discrete_feature = [feature for feature in numerical_features
                    if len(df_train[feature].unique()) < 25 and feature not in year_feature + ['Id']]
# Continuous
continuous_feature = [feature for feature in numerical_features if feature not in discrete_feature + year_feature + ['Id']]
# Categorical
categorical_features = [feature for feature in df_train.columns if df_train[feature].dtype == 'O']

# Print the count and unique values for each categorical feature
for feature in categorical_features:
    unique_values_count = df_train[feature].nunique()
    unique_values = df_train[feature].unique()

corr = df_train.drop(columns='Id').corr(numeric_only=True)
correlation=corr["SalePrice"].apply(abs).sort_values(ascending=False).reset_index()

df_train = df_train.drop((missing_data[missing_data['Total'] > 81]).index, axis=1)  # Remove columns with more than 81 NaN values
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)  # Remove rows with NaN in 'Electrical'
df_train = df_train.drop(correlation.iloc[21:, 0].values, axis=1)  # Remove columns with weak correlation

# Missing Data
name_of_coll = df_train.drop(columns="Id").select_dtypes(include=['number']).columns
for col in name_of_coll:
    nan_indices = df_train[col].isnull()  # Find the indices of NaN
    random_samples = df_train[col].dropna().sample(n=nan_indices.sum(), replace=True)  # Sample of the column without NaN
    df_train.loc[nan_indices, col] = random_samples.values
col_has_numbers = df_train.drop(columns="Id").select_dtypes(include=['number'])
col_has_numbers.isnull().sum().sort_values(ascending=False)
name_of_coll = df_train.drop(columns="Id").select_dtypes(include=['object']).columns
for col in name_of_coll:
    # Using mode()
    mode_for_coll = df_train[col].mode()[0]
    df_train[col].fillna(mode_for_coll, inplace=True)
col_has_objects = df_train.drop(columns="Id").select_dtypes(include=['object'])
col_has_objects.isnull().sum().sort_values(ascending=False)
for col in df_train.drop(columns="Id").select_dtypes(include=["number"]).columns:
    df_train = handle_outliers_iqr(df_train, col)

# Object (text) columns
obj_col = df_train.select_dtypes(include=['object']).columns
obj_col = pd.DataFrame(obj_col, columns=["text col"])
encoder = LabelEncoder()
for col in obj_col.values.flatten():
    df_train[col] = encoder.fit_transform(df_train[col])
df_train = df_train.drop(columns='MasVnrType')

# X and y data
X = df_train.iloc[:, :-1].values
y = df_train.iloc[:, -1:].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# StandardScaler
scaler = StandardScaler(copy=True, with_std=True)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Cross-validated scores
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(LinearRegression(), X_train, y_train, cv=cv)

# Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# Train and Test Scores
print(f"Train score: {model.score(X_train, y_train)}")
print(f"Test score: {model.score(X_test, y_test)}")
# Final Accuracy (R^2)
r2 = r2_score(y_test, y_pred)
print(f"Coefficient of Determination (R^2): {r2}")