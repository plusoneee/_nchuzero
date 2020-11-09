import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.preprocessing import Normalizer, RobustScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, RANSACRegressor, LogisticRegression
from sklearn.svm import SVC
from sklearn import tree

'''
Filter:
    mask = df['col_name'] == '?'
    df[mask]
Replace:
    df['col_name']replace('?', np.NaN)

Convert ignore NAN:
    df['col_name'] = df['col_name'].replace('?', np.NaN).astype(float)

Replace:
    df['col_name'].replace(('yes', 'no'), (1, 0)), inplace=True)
    
Drop:
    X = df.drop(['col_name'], axis=1)

Train and Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1000)
    
Cross Validation
    from sklearn.model_selection import
    scores = cross_val_score(lr, X, Y, cv=3, scoring='r2') #scoring: accuracy

Precision Score
    from sklearn.metrics import precision_score
    precision_score(Y_test, lr.predict(X_test))
    
Recall Score
    from sklearn.metrics import recall_score
    recall_score(Y_test, lr.predict(X_test))

f-measure Score
    from sklearn.metrics import fbeta_score, f1_score
    f1_score(Y_test, lr.predict(X_test))
    fbeta_score(Y_test, lr.predict(X_test), beta=1)
    # When beta=1, fbeta_score = s1_scroe
    # A beta less than 1 gives more importance to precision 
    # A value greater than 1 gives more importance to recall

ROC 
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(Y_test, Y_score)
    
AUC 
    from sklearn.metrics import auc
    auc(fpr, tpr)

'''

class FeatureSelection:

    @staticmethod
    def low_variance_filter_selection(df, threshold=1e-06):
        vt = VarianceThreshold(threshold)
        _X = df.to_numpy()
        X_ft = vt.fit_transform(_X)
        return df.loc[:, vt.get_support()]

    @staticmethod
    def select_k_best(df, label, k=10):
        kb_regr = SelectKBest(f_regression, k=k)
        X_b = kb_regr.fit_transform(df, label)
        cols = kb_regr.get_support(indices=True)
        return df.iloc[:, cols]

class Plot:

    @staticmethod
    def show_confusion_matrix(model, X_test, Y_test, title='Confusion Matrix'):
        confusion_matrix(y_true=Y_test, y_pred=model.predict(X_test))
        plot_confusion_matrix(model, X_test, Y_test, cmap=plt.cm.Blues)  # doctest: +SKIP
        plt.title(title)
        plt.show()

    @staticmethod
    def show_frequency_distribution_bar(df, col_name=None, title='Frequency Distribution', xLabel='Value', yLabel='Count'):
        _count = df[col_name].value_counts()
        sns.set(style="darkgrid")
        sns.barplot(_count.index, _count.values, alpha=0.9)
        plt.title(title)
        plt.ylabel(yLabel, fontsize=10)
        plt.xlabel(xLabel, fontsize=10)
        plt.show()

    @staticmethod
    def show_frequency_distribution_pie(df, col_name=None, title='Frequency Distribution', xLabel='Value', yLabel='Count'):
        labels = df[col_name].unique()
        counts = df[col_name].value_counts()
        sizes = [counts[var_cat] for var_cat in labels]
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)  # autopct is show the % on plot
        ax1.axis('equal')
        plt.show()

class Standardize:

    @staticmethod
    def rubust_scaler(df, col_name=None, q_range=(25, 75)):
        rbs = RobustScaler(quantile_range=q_range)
        # reshape np array to fit_transform
        col_array = df[col_name].to_numpy().reshape(-1, 1)
        scaled_col = rbs.fit_transform(col_array)

        # replace original data
        scaled_col = scaled_col.reshape(-1)
        df[col_name] = pd.Series(scaled_col)

        return df

    @staticmethod
    def normailizer(df, method='l2'):
        # methods: max, l1, l2
        no = Normalizer(norm=method)
        # reshape np array to fit_transform
        col_array = df.to_numpy()
        scaled_col = no.fit_transform(col_array)
        columns_name = df.columns
        print(','.join(columns_name))
        df = pd.DataFrame(scaled_col, columns=columns_name)
        return df

    @staticmethod
    def standard_scaler(df, col_name=None):
        ss = StandardScaler()

        # reshape np array to fit_transform
        col_array = df[col_name].to_numpy().reshape(-1, 1)
        scaled_col = ss.fit_transform(col_array)

        # replace original data
        scaled_col = scaled_col.reshape(-1)
        df[col_name] = pd.Series(scaled_col)

        return df

class ValueReplacer:

    @staticmethod
    def replace_value(df, col_name, new_value, old_value):
        mask = df[col_name] == old_value
        df[col_name][mask] = new_value
        return df

    @staticmethod
    def tokenizer_by_column(df, col_name=None):
        valus_unique_list = df[col_name].unique()
        count = 0
        record_dict = dict()
        for value in valus_unique_list:
            mask = (df[col_name] == value)
            df[col_name][mask] = count
            record_dict[value] = count
            count += 1

        ValueReplacer.export(col_name, record_dict)
        return df

    @staticmethod
    def fill_missing_value(df, col_name=None, method='mean'):
        if method == 'mean':
            df[col_name] = df[col_name].fillna(df[col_name].mean())
        elif method == 'median':
            df[col_name] = df[col_name].fillna(df[col_name].median())
        elif method == 'most_frequent':
            df[col_name].fillna(df[col_name].mode()[0])

        return df

    @staticmethod
    def fill_missing_value_by_imputer(df, col_name=None, method='mean'):
        imputer = SimpleImputer(missing_values=np.nan, strategy=method)
        # turn to np array to use reshape
        df_np = np.array(df[col_name])
        missing_new_value = imputer.fit_transform(df_np.reshape(-1, 1)).squeeze()
        df[col_name] = pd.Series(missing_new_value)
        return df

    @staticmethod
    def export(title, content):
        file_path = title +'.json'
        with open('./'+file_path, 'w') as f:
            json.dump(content, f)


class Model:

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def fit_linear_regression(self):
        lr = LinearRegression()
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        lr.fit(X_train, y_train)
        print('Linear Regression Score:', lr.score(X_test, y_test))
        print('Coefficient:', lr.coef_)
        return lr

    def fit_ransac_regressor(self, random_state=0):
        # X, y = make_regression(n_samples = 200, n_features = 2, noise = 4.0, random_state = 0)
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        rs = RANSACRegressor(random_state).fit(X_train, y_train)
        print('RANSACRegressor Score:', rs.score(X_test, y_test))
        return rs

    def fit_decision_tree(self, max_depth=4):
        dt = tree.DecisionTreeClassifier(max_depth)
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        dt.fit(X_train, y_train)
        print('Decision Tree Classifier Score:', dt.score(X_test, y_test))
        return dt

    def fit_logistic_regression(self):
        lg = LogisticRegression()
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        lg.fit(X_train, y_train)
        print('Logistic regression score: %.3f' % lg.score(X_test, y_test))
        return lg

    def fit_svm(self, ovr=False):
        # one-versus-rest
        if ovr:
            svc = SVC(decision_function_shape='ovr')
        else:
            svc = SVC(kernel='linear')
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        svc.fit(X_train, y_train)
        # get support vectors
        # clf.support_vectors_
        # get indices of support vectors
        # clf.support_
        # get number of support vectors for each class
        # clf.n_support_
        return svc


