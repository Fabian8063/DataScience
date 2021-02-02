import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import interactive
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

list_important_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingMovies', 'Contract', 'PaymentMethod', 'tenure', 'MonthlyCharges',
                           'TotalCharges']
list_imp_cat_col = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingMovies', 'Contract', 'PaymentMethod']
list_imp_num_col = ['tenure', 'MonthlyCharges', 'TotalCharges']

def prepare_data():
    """
    Loads the data into a DataFrame and transforms it.

    Returns
    -------
    df: DataFrame
        DataFrame with transformed data.
    """
    df = pd.read_csv('Kunden.csv')
    df = df[df['TotalCharges'] != ' ']  # remove 11 customers with no total charges
    df = df.reset_index(drop=True)
    df = df.astype({'TotalCharges': float})
    df = df.drop(columns='customerID')
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df


def _split_data_churn_no_churn(df):
    """
    Splits customers into two DataFrames. One for customers that churned and one for customers that did not.

    Parameters
    ----------
    df: DataFrame
        The DataFrame with input data.

    Returns
    -------
    df_churn: DataFrame
        DataFrame with customers that churned.
    df_no_churn: DataFrame
        DataFrame with customers that did not churn.
    """
    df_churn = df[df['Churn'] == 1].reset_index(drop=True)
    df_no_churn = df[df['Churn'] == 0].reset_index(drop=True)
    return df_churn, df_no_churn


def analyse_data(df):
    """
    Analyses the data and visualises the features.

    Parameters
    ----------
    df: DataFrame
        The DataFrame with input data.
    """
    list_cat_col = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    list_num_col = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    df_churn, df_no_churn = _split_data_churn_no_churn(df)
    # numerical features
    print('Churner')
    for i in range(len(list_num_col)):
        print(list_num_col[i])
        print(df_churn[list_num_col[i]].mean())  # also check median, mode
    print('No Churner')
    for i in range(len(list_num_col)):
        print(list_num_col[i])
        print(df_no_churn[list_num_col[i]].mean())  # also check median, mode
    # categorical features
    for i in range(len(list_cat_col)):
        plt.figure(i)
        sns.countplot(df_no_churn[list_cat_col[i]], color='blue')

    for i in range(len(list_cat_col)):
        plt.figure(i + 15)
        sns.countplot(df_churn[list_cat_col[i]], color='red')
    interactive(False)
    plt.show()


def build_prediction(df, importance):
    """
    Encodes the categorical features and builds an estimator for churn prediction.

    Parameters
    ----------
    df: DataFrame
        The DataFrame with input data.
    importance: boolean
        Whether feature importance should be printed or not.

    Returns
    -------
    Y_pred: np.array
        Array with churn prediction.
    Y_test: np.array
        Array with true values for target column.
    """
    X = df.drop(columns='Churn')
    Y = df['Churn']
    enc = OrdinalEncoder()
    model = DecisionTreeClassifier(max_depth=8)
    categorical_features = list_imp_cat_col
    categorical_transformer = Pipeline(steps=[('enc', enc)])
    preprocessor = ColumnTransformer(transformers=[('cat_trans', categorical_transformer, categorical_features)])
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('tree', model)])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, stratify=df['Churn'])
    pipe.fit(X_train, Y_train)
    if (importance):
        feat_imp = pipe.named_steps['tree'].feature_importances_
        list_col_names = df.columns.to_list()
        list_col_names.remove('Churn')
        feat_imp_col = list(zip(list_col_names, feat_imp))
        print(feat_imp_col)
    Y_pred = pipe.predict(X_test) # use predict_proba() for prediction of probabilities
    return Y_pred, Y_test


def validate_prediction(Y_pred, Y_test):
    """
    Calculates the accuracy of the prediction.

    Parameters
    ----------
    Y_pred: np.array
        Array with churn prediction.
    Y_test: np.array
        Array with true values for target column.

    Returns
    -------
    score: float
        Accuracy of the prediction.
    """
    score = accuracy_score(Y_test, Y_pred)
    return score


if __name__ == '__main__':
    df = prepare_data()
    # uncomment for data visualisation
    # analyse_data(df)
    df = df[list_important_features + ['Churn']]
    # since train_test_split is random, calculate mean validation score
    mean_score = 0
    iterations = 10
    for i in range(iterations):
        Y_pred, Y_test = build_prediction(df, importance=False)
        mean_score = mean_score + validate_prediction(Y_pred, Y_test)
    mean_score = mean_score / iterations
    print("Durchschnittliche Genauigkeit nach 10 Durchläufen: {}".format(mean_score))
