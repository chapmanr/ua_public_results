import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

import os


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

from sklearn.metrics            import accuracy_score, make_scorer
from sklearn.model_selection    import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import time
import joblib
from sklearn.model_selection import KFold
import time
from scipy.stats import randint
from sklearn.metrics import classification_report


SEED = 42

#ImportError: cannot import name 'is_scalar_nan' from 'sklearn.utils'
#from embedding_encoder import EmbeddingEncoder doesn't work



def load_raw_data():
    
    X_train = pd.read_csv('clean_data/X_train.csv', index_col='ID')
    y_train = pd.read_csv('clean_data/y_train.csv', index_col='ID')
    X_pred  = pd.read_csv('clean_data/X_test.csv', index_col='ID')

    return X_train, y_train, X_pred

def basic_preprocessing(X, basic_only=False):
    X["brand"] = X["brand"].str.strip()
    X['brand'] = X['brand'].str.replace('M', 'm')
    X['brand'] = X['brand'].str.replace('marca', '')
    X["brand"] = X["brand"].str.strip()

    #SKU is an ID - therefore categorical not a "real" number
    X['sku'] = X['sku'].apply(lambda x: str(x))
    
    
    X[['new_pvp', 'discount']] = X['new_pvp (discount)'].str.split(' ', expand=True)
    X.drop(['new_pvp (discount)'], axis=1, inplace=True)

    X['oldpvp']     = X['oldpvp'].apply(lambda x: str(x).replace(',', '.'))
    #Prices should be numbers
    X['oldpvp']     = X['oldpvp'].apply(lambda x: float(x))

   
    X['new_pvp']    = X['new_pvp'].apply(lambda x: str(x).replace(',', '.'))
    #Prices should be numbers
    X['new_pvp']    = X['new_pvp'].apply(lambda x: float(x))


    X['discount'] = X['discount'].apply(lambda x: str(x).replace('%', ''))
    X['discount'] = X['discount'].apply(lambda x: str(x).replace('(', ''))
    X['discount'] = X['discount'].apply(lambda x: str(x).replace(')', ''))
    X['discount'] = X['discount'].apply(lambda x: '0.'+x if len(x) == 2 else x)

    X['discount'] = X['discount'].apply(lambda x: float(x))
    X["discount"] = X["discount"].apply(lambda x: x if x <= 1.0 else x / 100)

    #Weight should be a number - but I'm not sure how useful it is...
    X['weight (g)'] = X['weight (g)'].apply(lambda x: str(x).replace(' ', ''))
    most_frequ_weight = X['weight (g)'].mode()
    most_frequ_weight = most_frequ_weight[0]

    X['weight (g)'] = X['weight (g)'].apply(lambda x: int(x) if x != '' and x != 'nan' else int(most_frequ_weight))


    X['expiring_date'] = X['expiring_date'].apply(lambda x: str(x).replace('/', '-'))
    X['labelling_date'] = X['labelling_date'].apply(lambda x: str(x).replace('/', '-'))
    
    
    X['expiring_date'] = pd.to_datetime(X['expiring_date'], format='%d-%m-%Y')
    X['expiring_day'] = X['expiring_date'].dt.dayofweek
    X['expiring_day'] = X['expiring_day'].astype(str)

    X["idstore"] = X["idstore"].apply(str)       

    X['labelling_date'] = pd.to_datetime(X['labelling_date'], format='mixed')
    X["labelling_day"] = X["labelling_date"].dt.dayofweek  
    X["labelling_day"] = X["labelling_day"].astype(str)


    X['duration_days'] = pd.to_datetime(X['expiring_date']) - pd.to_datetime(X['labelling_date'])
    X['duration_days'] =  X['duration_days'].dt.total_seconds() / (24 * 60 * 60)

    # calculate total cost from profit and margin
    X_cost = ((100 - X['Margin (%)'].to_numpy()) / X['Margin (%)'].to_numpy()) * X['Profit (€)'].to_numpy()
    X.loc[:, ('Cost (€)')] = X_cost

    X.drop(['labelqty'], axis=1, inplace=True)
    X.drop(['labelling_date'], axis=1, inplace=True)
    X.drop(['expiring_date'], axis=1, inplace=True)
    
    #Which one of these buggers is now irrelevent?
    #X.drop(['new_pvp', 'oldpvp'], axis=1, inplace=True)

    X.drop(['Margin (%)'], axis=1, inplace=True)
    
    return X


def common_stats(X, y, log):

    log.title('Raw Data Exploration')
    
    log.subtitle('X')
    log.table(X.head())
    
    log.subtitle('y')
    log.table(y.head())

    log.subtitle('X info')
    
    buffer = io.StringIO()
    X.info(buf=buffer, verbose=True)
    s = buffer.getvalue()
    log.text(s)

    log.subtitle('y info')
    buffer = io.StringIO()
    y.info(buf=buffer, verbose=True)
    s = buffer.getvalue()
    log.text(s)

    log.subtitle('y value class counts')
    log.table(y.value_counts(normalize=True))
    
    log.subtitle("X stats data")

    log.table(X.describe())

    log.subtitle("X missing values")
    log.table(X.isnull().sum())

from sklearn import metrics

#X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y,random_state=42,test_size=0.3) 
#print(X_train.shape, X_test.shape)

def plot_roc_curve(clf, est, fold, X_test_split, y_test_split, show=False):
    metrics.RocCurveDisplay.from_estimator(clf, X_test_split, y_test_split[['sold']], name=est)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    fig_filename = 'results/ROC_'+ est + '_fold_' + str(fold) + '.png'
    plt.savefig(fig_filename)
    if show:
        plt.show()
    return fig_filename

def plot_precision_recall_curve(clf, est, fold, X_test_split, y_test_split, show=False):
    metrics.PrecisionRecallDisplay.from_estimator(clf, X_test_split, y_test_split[['sold']], name=est)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    fig_filename = 'results/PRC_'+ est + '_fold_' + str(fold) + '.png'
    plt.savefig(fig_filename)
    if show:
        plt.show()
    return fig_filename



def hyperparameter_tune(base_model, parameters, n_iter, kfold, X, y):
    start_time = time.time()
    
    # Arrange data into folds with approx equal proportion of classes within each fold
    k = StratifiedKFold(n_splits=kfold, shuffle=False)
    
    optimal_model = RandomizedSearchCV(base_model,
                            param_distributions=parameters,
                            n_iter=n_iter,
                            cv=k,
                            n_jobs=-1,
                            random_state=SEED)
    
    optimal_model.fit(X, y)
    
    stop_time = time.time()

    scores = cross_val_score(optimal_model, X, y, cv=k, scoring="accuracy")
    
    print("Elapsed Time:", time.strftime("%H:%M:%S", time.gmtime(stop_time - start_time)))
    print("====================")
    print("Cross Val Mean: {:.3f}, Cross Val Stdev: {:.3f}".format(scores.mean(), scores.std()))
    print("Best Score: {:.3f}".format(optimal_model.best_score_))
    print("Best Parameters: {}".format(optimal_model.best_params_))
    
    return optimal_model.best_params_, optimal_model.best_score_


def display_numerical_histograms(X, log, path, name="numerical_histograms", bins=50):
    plt.axes().clear()

    numerical_columns = X.select_dtypes(include='number').columns.tolist()
    X[numerical_columns].hist(bins=bins, figsize=(20,15))

    plt.savefig(path + name + '.png')    
    log.subtitle(name)
    log.img(name + '.png')


def default_pipeline_process(estimator, X_train_s, y_s):
    numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scale', MinMaxScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('one-hot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_features = X_train_s.select_dtypes(include='number').columns.tolist()
    categorical_features = X_train_s.select_dtypes(include='object').columns.tolist()

    full_processor = ColumnTransformer(transformers=[
        ('number', numeric_pipeline, numerical_features),
        ('category', categorical_pipeline, categorical_features)
    ])    

    proc_pipeline = Pipeline(steps=[
    ('preprocess', full_processor),
    ('model', estimator)    
        ])
    clf = proc_pipeline.fit(X_train_s, y_s['sold'])
    return clf

def performance_pipeline_process(estimator, X_train_s, y_s):
    numeric_pipeline = Pipeline(steps=[
    ('impute', KNNImputer(n_neighbors=5)),
    ('transform', PowerTransformer(method='yeo-johnson', standardize=False)),
    ('scale', MinMaxScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('impute', KNNImputer(n_neighbors=5)),
        ('one-hot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_features = X_train_s.select_dtypes(include='number').columns.tolist()
    categorical_features = X_train_s.select_dtypes(include='object').columns.tolist()

    full_processor = ColumnTransformer(transformers=[
        ('number', numeric_pipeline, numerical_features),
        ('category', categorical_pipeline, categorical_features)
    ])    

    proc_pipeline = Pipeline(steps=[
    ('preprocess', full_processor),
    ('model', estimator)    
        ])
    clf = proc_pipeline.fit(X_train_s, y_s['sold'])
    return clf

def get_estimator(est):
    estimator = None

    match est:
        case "sgd":
            estimator = SGDClassifier(max_iter=10000, tol=1e-3)
        case "tree":
            estimator = DecisionTreeClassifier(random_state=0)
        case "forest":
            estimator = RandomForestClassifier(random_state=0, n_jobs=-1)#speedy up
        case "xgboost":
            estimator = XGBClassifier(random_state=0)
        case "knn":
            estimator = KNeighborsClassifier()
    return estimator

def run_all_default(X_train, y, X_test, log_to):
    scores = {}
    best = True

    all_est = ["sgd", "tree", "forest", "xgboost", "knn"]#ann in a bit
    all_start = time.time()
    for est in all_est:
       
        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        scores[est] = {}
        

        scores[est] = {}
        scores[est]['train'] = []
        scores[est]['test'] = []

        start_train = time.time()

        estimator = get_estimator(est)
        log_to.bold("Training " + est)
        print(str(estimator))

        for fold, (train_index, test_index) in enumerate(kf.split(X_train)):
            scores[est][fold] = {}
            X_train_split, X_test_split = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_split, y_test_split = y.iloc[train_index], y.iloc[test_index]

            clf = default_pipeline_process(estimator, X_train_split, y_train_split)                

            print(str(type(clf)))

            train_score =  clf.score(X_train_split, y_train_split[['sold']])
            scores[est]['train'].append(train_score)

            log_to.bold('Training ' + str(fold)  +  est + '_best' + str(best) + ' Train - Test Split Score:' + str(train_score))
            print('Training ' + str(fold)  +  est + '_best' + str(best) + ' Train - Test Split Score:' , train_score)

            y_pred = clf.predict(X_train_split)

            report = classification_report(y_train_split[['sold']], y_pred)
            print(report)
            log_to.text(report)

            with open('results/classification_report_' + est + '_best' + str(best) + '_training.txt', 'w') as f:
                f.write(report)                

            test_score = clf.score(X_test_split, y_test_split[['sold']])

            scores[est]['test'].append(test_score)

            print('Test' +  est + '_best' + str(best) + ' Train - Test Split Score:', test_score)
            log_to.bold('Test' +  est + '_best' + str(best) + ' Train - Test Split Score:' + str(test_score))


            y_pred = clf.predict(X_test_split)
            report = classification_report(y_test_split[['sold']], y_pred)
            print(report)
            log_to.text(report)

            with open('results/classification_report_' + est + '_best' + str(best) + '_test.txt', 'w') as f:
                f.write(report)

            roc_filename = plot_roc_curve(clf, est, fold, X_test_split, y_test_split)
            log_to.img(roc_filename)

            prc_filename = plot_precision_recall_curve(clf, est, best, X_test_split, y_test_split)
            log_to.img(prc_filename)

        joblib.dump(clf, 'models/model_' + str(fold) + est + '_best' + str(best) + '_defaults_preprocessed.pkl')

        with open('results/scores_' + est + '_best' + str(best) + '_defaults_preprocessed.txt', 'w') as f:
            f.write(str(scores))
        

        end_train = time.time()
        print('Training time : for = ' + est, end_train - start_train)
        log_to.bold('Training time : for = ' + est + str(end_train - start_train))

        y_actual_pred = clf.predict(X_test)
        pred_df = pd.DataFrame(y_actual_pred.astype(int), columns=['sold'], index=X_test.index)
        pred_df.to_csv('results/y_test_' + est + '_best' + str(best) + 'defaults.csv') 

    print('Total time : for = ' + est, time.time() - all_start)
    log_to.bold('Total time : for = ' + est + str(time.time() - all_start))



















def apply_numerical_pipeline(X_DataFrame, log, to_remove=['idstore']):
    ''' Primarily used to evaluate the numerical data and to transform it for display purposes'''        
    
    numeric_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('transform', PowerTransformer(method='yeo-johnson', standardize=False)),
        ('scale', MinMaxScaler())            
    ])

    numerical_features = X_DataFrame.select_dtypes(include='number').columns.tolist()

    # not the id store though
    for label in to_remove:
        if label in numerical_features:
            numerical_features.remove(label)

    num_processor = ColumnTransformer(transformers=[('number', numeric_pipeline, numerical_features)])
    num_processor.fit(X_DataFrame)    
    X_transformed = num_processor.transform(X_DataFrame)   

    X_transformed = pd.DataFrame(X_transformed, columns=numerical_features)
    return X_transformed



def random_search(X_train, y_train, classifier, param_grid, log):
    scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}

    # Setting refit='AUC', refits an estimator on the whole dataset with the
    # parameter setting that has the best cross-validated AUC score.
    # That estimator is made available at ``gs.best_estimator_`` along with
    # parameters like ``gs.best_score_``, ``gs.best_params_`` and
    # ``gs.best_index_``

    
    n_iter = 10
    kfold = 5

    # Arrange data into folds with approx equal proportion of classes within each fold
    k = StratifiedKFold(n_splits=kfold, shuffle=False)
        
    if isinstance(classifier, KNeighborsClassifier):
        gs = RandomizedSearchCV(classifier, param_grid, cv=k , n_iter = 15)
    else:
        gs = RandomizedSearchCV(classifier,
                            param_distributions=param_grid,
                            n_iter=n_iter,
                            cv=k,
                            n_jobs=-1,
                            random_state=42,
                            scoring=scoring,
                            refit="AUC",
                            return_train_score=True,
                            verbose=3
                            )
    
    my_pipeline = Pipeline(steps=[
        ('preprocess', default_pipeline_process(X_train, log)),
        ('model', gs)
    ])

    start = time.time()


    clf = my_pipeline.fit(X_train, y_train['sold'])
    
    finish = time.time()    
    
    log.bold("Time taken to train : " + str(finish - start))

    return gs, clf, my_pipeline


def grid_search(X_train, y_train, classifier, param_grid, log):
    scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}

    # Setting refit='AUC', refits an estimator on the whole dataset with the
    # parameter setting that has the best cross-validated AUC score.
    # That estimator is made available at ``gs.best_estimator_`` along with
    # parameters like ``gs.best_score_``, ``gs.best_params_`` and
    # ``gs.best_index_``
    gs = GridSearchCV(
        classifier,
        param_grid=param_grid,
        scoring=scoring,
        refit="AUC",
        n_jobs=2,
        return_train_score=True,
    )

    my_pipeline = Pipeline(steps=[
        ('preprocess', default_pipeline_process(classifier, X_train, log)),
        ('model', gs)
    ])
    start = time.time()


    clf = my_pipeline.fit(X_train, y_train['sold'])
    
    finish = time.time()    
    
    log.bold("Time taken to train : " + str(finish - start))

    return gs, clf, my_pipeline

def plot_model_feature_performance(clf, tosave_as='feature_importance.png'):    
    importance = clf['model'].feature_importances_.argsort()[::-1][0:20]
    importance_values=clf['model'].feature_importances_[importance]
    features= clf['preprocess'].get_feature_names_out()[importance]
    
    plt.bar(features, importance_values)
    plt.xticks(rotation=90)
    plt.savefig(tosave_as)


def display_results(results, scoring, log, x_axis, title="GridSearchCV evaluating using multiple scorers simultaneously"):
    plt.figure(figsize=(13, 13))
    plt.title(title, fontsize=16)

    #plt.xlabel("min_samples_split")
    plt.ylabel("Score")

    ax = plt.gca()
    #ax.set_xlim(0, 402)
    #ax.set_ylim(0.6, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results["param_min_samples_split"].data, dtype=float)


    for scorer, color in zip(sorted(scoring), ["g", "k"]):
        for sample, style in (("train", "--"), ("test", "-")):
            sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
            sample_score_std = results["std_%s_%s" % (sample, scorer)]
            ax.fill_between(
                X_axis,
                sample_score_mean - sample_score_std,
                sample_score_mean + sample_score_std,
                alpha=0.1 if sample == "test" else 0,
                color=color,
            )
            ax.plot(
                X_axis,
                sample_score_mean,
                style,
                color=color,
                alpha=1 if sample == "test" else 0.6,
                label="%s (%s)" % (scorer, sample),
            )

        best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
        best_score = results["mean_test_%s" % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot(
            [
                X_axis[best_index],
            ]
            * 2,
            [0, best_score],
            linestyle="-.",
            color=color,
            marker="x",
            markeredgewidth=3,
            ms=8,
        )

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.savefig('results/grid_search.png')


def raw_data_info(path:str):
    X_train, y_train, X_test = load_raw_data(path)
    print(X_train.info())
    print(y_train.info())
    print(X_test.info())


def raw_data_describe(path:str):
    X_train, y_train, X_test = load_raw_data(path)
    print(X_train.describe())
    print(y_train.describe())
    print(X_test.describe())

def raw_data_shape(path:str):
    X_train, y_train, X_test = load_raw_data(path)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)


def data_reformat(X):
    
    if "brand" in X.columns:
        X["brand"] = X["brand"].str.strip()
        X['brand'] = X['brand'].str.replace('M', 'm')
        X['brand'] = X['brand'].str.replace('marca', '')
        X["brand"] = X["brand"].str.strip()

    if 'new_pvp (discount)' in X.columns:   
        X[['new_pvp', 'discount']] = X['new_pvp (discount)'].str.split(' ', expand=True)
        X.drop(['new_pvp (discount)'], axis=1, inplace=True)

    if 'oldpvp' in X.columns:
        X['oldpvp'] = X['oldpvp'].apply(lambda x: str(x).replace(',', '.'))
    
    if 'new_pvp' in X.columns:
        X['new_pvp'] = X['new_pvp'].apply(lambda x: str(x).replace(',', '.'))
    
    if 'discount' in X.columns:
        X['discount'] = X['discount'].apply(lambda x: str(x).replace('%', ''))
        X['discount'] = X['discount'].apply(lambda x: str(x).replace('(', ''))
        X['discount'] = X['discount'].apply(lambda x: str(x).replace(')', ''))
        X['discount'] = X['discount'].apply(lambda x: '0.'+ x if len(x) == 2 else x)

    if 'labelqty' in X.columns:
        X.drop(['labelqty'], axis=1, inplace=True)
    
    return X


def raw_data_reformat(path:str):
    X_train, y_train, X_test = load_raw_data(path)
    X_train = data_reformat(X_train)
    X_test = data_reformat(X_test)
    return X_train, y_train, X_test


def raw_data_reformat_save():
    X_train, y_train, X_test = raw_data_reformat()
    X_train.to_csv('X_train_reformat.csv')
    X_test.to_csv('X_test_reformat.csv')


def load_data_reformat():
    X_train = pd.read_csv('X_train_reformat.csv', index_col='ID')
    y_train = pd.read_csv('y_train.csv', index_col='ID')
    X_test = pd.read_csv('X_test_reformat.csv', index_col='ID')
    return X_train, y_train, X_test

def load_data_reformat_info():
    X_train, y_train, X_test = load_data_reformat()
    print(X_train.info())
    print(y_train.info())
    print(X_test.info())

def date_time_refactor(X):
    X['expiring_date'] = X['expiring_date'].apply(lambda x: str(x).replace('/', '-'))
    X['expiring_date'] = pd.to_datetime(X['expiring_date'], format='%d-%m-%Y')
    X['expiring_day'] = X['expiring_date'].dt.dayofweek

    X['labelling_date'] = X['labelling_date'].apply(lambda x: str(x).replace('/', '-'))
    X['labelling_date'] = pd.to_datetime(X['labelling_date'], format='mixed')
    X["labelling_day"] = X["labelling_date"].dt.dayofweek  

    X['duration_days'] = pd.to_datetime(X['expiring_date']) - pd.to_datetime(X['labelling_date'])
    X['duration_days'] =  X['duration_days'].dt.total_seconds() / (24 * 60 * 60)
  
  
    X.drop(['labelling_date'], axis=1, inplace=True)
    X.drop(['expiring_date'], axis=1, inplace=True)

    return X

    


def get_categorical_features(X):
    return X.select_dtypes(include=['object']).columns

def get_numerical_features(X):
    return X.select_dtypes(include=['number']).columns


def get_categorical_data(X):
    return X[get_categorical_features(X)]

def get_numerical_data(X):
    return X[get_numerical_features(X)]


def show_correlations(data_set=0, show_graphs = True):

    X_train, y_train, X_test = load_raw_data()

    if data_set == 1:
        X_train, y_train, X_test = load_data_reformat()
    elif data_set == 2:
        X_train, y_train, X_test = load_data_reformat_discount()
    elif data_set == 3:
        X_train, y_train, X_test = load_data_reformat_date()
    elif data_set == 4:
        X_train, y_train, X_test = load_data_reformat_impute()
    elif data_set == 5:
        X_train, y_train, X_test = load_data_reformat_weight()
    elif data_set == 6:
        X_train, y_train, X_test = load_data_reformat_transform()



    X_train_num = get_numerical_data(X_train)
    print("========Numerical Null Values========")
    print(str(X_train_num.isnull().mean()))

    print("========Categorical Null Values========")    
    X_train_cat = get_categorical_data(X_train)
    print(str(X_train_cat.isnull().mean()))

    print("========Correlation========")

    X_train_num.hist(bins=50, figsize=(20,15))
    if show_graphs:
        plt.show()    

    correlation_matrix = pd.concat([X_train_num, y_train], axis=1).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')

    if show_graphs:
        plt.show()    

def compare_discount():
    X_train, y_train, X_test = load_data_reformat()
    print("=== Discount ===")
    #print(X_train['discount'].value_counts(normalize=True))
    print(X_train['discount'].unique())

    print("=== New PVP ===")
    #print(X_train['new_pvp'].value_counts(normalize=True))
    print(str(X_train['new_pvp'].unique()))

    print("=== Old PVP ===")
    #print(X_train['oldpvp'].value_counts(normalize=True))
    print(X_train['oldpvp'].unique())


    print("=== Old PVP ===")
    f_old_pvp = X_train['oldpvp'].astype(float)
    print(f_old_pvp.describe())

    print("=== New PVP ===")
    f_new_pvp = X_train['new_pvp'].astype(float)
    print(f_new_pvp.describe())

    print("=== Discount ===")
    f_discount = X_train['discount'].astype(float)
    print(f_discount.describe())

    print("=== Check Discount ===")
    check_discount = ((f_old_pvp - f_new_pvp) / f_old_pvp) 
    print(check_discount.describe())

def replace_discount(X):

    f_old_pvp = X['oldpvp'].astype(float)
    f_new_pvp = X['new_pvp'].astype(float)
    
    check_discount = ((f_old_pvp - f_new_pvp) / f_old_pvp)
    
    X['discount'] = check_discount

    return X

def replace_discount_save():
    X_train, y_train, X_test = load_data_reformat()
    X_train = replace_discount(X_train)
    X_test = replace_discount(X_test)
    X_train.to_csv('X_train_reformat_discount.csv')
    X_test.to_csv('X_test_reformat_discount.csv')

def load_data_reformat_discount():
    X_train = pd.read_csv('X_train_reformat_discount.csv', index_col='ID')
    y_train = pd.read_csv('y_train.csv', index_col='ID')
    X_test = pd.read_csv('X_test_reformat_discount.csv', index_col='ID')
    return X_train, y_train, X_test

def date_time_refactor_save():
    X_train, y_train, X_test = load_data_reformat_discount()
    X_train = date_time_refactor(X_train)
    X_test = date_time_refactor(X_test)



    X_train.to_csv('X_train_reformat_date.csv')
    X_test.to_csv('X_test_reformat_date.csv')

def load_data_reformat_date():
    X_train = pd.read_csv('X_train_reformat_date.csv', index_col='ID')
    y_train = pd.read_csv('y_train.csv', index_col='ID')
    X_test = pd.read_csv('X_test_reformat_date.csv', index_col='ID')
    return X_train, y_train, X_test

from sklearn.impute import SimpleImputer

def ImputeValues(X):  
    numerical_features      = X.select_dtypes(include='number').columns.tolist()
    categorical_features    = X.select_dtypes(include='object').columns.tolist() 
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])
    num_imputer = SimpleImputer(strategy='median')
    X[numerical_features] = num_imputer.fit_transform(X[numerical_features])
    return X

def ImputeValues_save():
    X_train, y_train, X_test = load_data_reformat_date()
    X_train = ImputeValues(X_train)
    X_test = ImputeValues(X_test)
    X_train.to_csv('X_train_reformat_impute.csv')
    X_test.to_csv('X_test_reformat_impute.csv')

def load_data_reformat_impute():
    X_train = pd.read_csv('X_train_reformat_impute.csv', index_col='ID')
    y_train = pd.read_csv('y_train.csv', index_col='ID')
    X_test = pd.read_csv('X_test_reformat_impute.csv', index_col='ID')
    return X_train, y_train, X_test

def investigate_weight(X):
    most_frequ_weight = X['weight (g)'].mode()
    most_frequ_weight = most_frequ_weight[0]
  
    X['weight (g)'] = X['weight (g)'].apply(lambda x: str(x).replace(' ', most_frequ_weight))
    
    unique_weight = X['weight (g)'].unique()
    
    print(str(unique_weight))

    X['weight (g)'] = X['weight (g)'].apply(lambda x: str(x).replace('g', ''))
    X['weight (g)'] = X['weight (g)'].apply(lambda x: str(x).replace('kg', '000'))
    X['weight (g)'] = X['weight (g)'].apply(lambda x: str(x).replace('ml', ''))
    X['weight (g)'] = X['weight (g)'].apply(lambda x: str(x).replace('l', '000'))

    return X

def investigate_weight_save():
    X_train, y_train, X_test = load_data_reformat_impute()
    X_train = investigate_weight(X_train)
    X_test = investigate_weight(X_test)
    X_train.to_csv('X_train_reformat_weight.csv')
    X_test.to_csv('X_test_reformat_weight.csv')

def load_data_reformat_weight():
    X_train = pd.read_csv('X_train_reformat_weight.csv', index_col='ID')
    y_train = pd.read_csv('y_train.csv', index_col='ID')
    X_test = pd.read_csv('X_test_reformat_weight.csv', index_col='ID')
    return X_train, y_train, X_test



def run_all():
    raw_data_info()
    raw_data_reformat_save()
    load_data_reformat_info()

    X_train, y_train, X_test = load_data_reformat()

    print(str(get_categorical_features(X_train)))
    print(str(get_numerical_features(X_train)))

    show_correlations(1, False)

    compare_discount()
    replace_discount_save()

    X_train, y_train, X_test = load_data_reformat_discount()

    show_correlations(2, False)

    date_time_refactor_save()
    X_train, y_train, X_test = load_data_reformat_date()

    show_correlations(3, False)

    #Now imputation
    ImputeValues_save()
    X_train, y_train, X_test = load_data_reformat_impute()
    show_correlations(4, True)

    X_train, y_train, X_test = load_data_reformat_impute()
    investigate_weight(X_train)

    investigate_weight_save()

    X_train, y_train, X_test = load_data_reformat_weight()
    show_correlations(5, True)


from sklearn.preprocessing import PowerTransformer

def transform_data(X):
    p_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    numerical_columns = X.select_dtypes(include='number').columns.tolist()
    new_features = p_transformer.fit_transform(X[numerical_columns])
    df_new_features = pd.DataFrame(new_features, columns=numerical_columns)
    X[numerical_columns] = df_new_features
    return X

def transform_data_save():
    X_train, y_train, X_test = load_data_reformat_impute()
    X_train = transform_data(X_train)
    X_test = transform_data(X_test)
    X_train.to_csv('X_train_reformat_transform.csv')
    X_test.to_csv('X_test_reformat_transform.csv')

def load_data_reformat_transform():
    X_train = pd.read_csv('X_train_reformat_transform.csv', index_col='ID')
    y_train = pd.read_csv('y_train.csv', index_col='ID')
    X_test = pd.read_csv('X_test_reformat_transform.csv', index_col='ID')
    return X_train, y_train, X_test

from    sklearn.model_selection import train_test_split

def load_split_data(data_set=0):
    X_train, y_train, X_test = load_raw_data()

    if data_set == 1:
        X_train, y_train, X_test = load_data_reformat()
    elif data_set == 2:
        X_train, y_train, X_test = load_data_reformat_discount()
    elif data_set == 3:
        X_train, y_train, X_test = load_data_reformat_date()
    elif data_set == 4:
        X_train, y_train, X_test = load_data_reformat_impute()
    elif data_set == 5:
        X_train, y_train, X_test = load_data_reformat_weight()
    elif data_set == 6:
        X_train, y_train, X_test = load_data_reformat_transform()

    X, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42, test_size=0.3)
    return X, X_test, y_train, y_test

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import pickle
import time



def random_forest(X_train, y_train, X_test):
    now = time.time()    
    model = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100, n_jobs=1)
    model.fit(X_train, y_train)
    
    print("Time taken to train : ", time.time() - now)

    filename = 'random_forest.model'
    pickle.dump(model, open(filename, 'wb'))
    
    
    y_pred = model.predict(X_test)
    return y_pred

#Now apply min max scaler
from sklearn.preprocessing import MinMaxScaler

def scale_data(X):
    scaler = MinMaxScaler()
    numerical_columns = X.select_dtypes(include='number').columns.tolist()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    return X




