import pandas as pd
import numpy as np

from sklearn.model_selection import (train_test_split,RandomizedSearchCV)

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (SelectKBest,f_regression)

from sklearn.metrics import r2_score
from scipy.stats import uniform
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE


# =========================
# PREPROCESSING
# =========================

def split_scalar(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.30,
        random_state=0
    )

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, Y_train, Y_test

# selectK
def selectkbest(X, Y, n):

    selector = SelectKBest(score_func=f_regression, k=n)
    return selector.fit_transform(X, Y)


# PCA
def pca_features(X, n):

    X = pd.get_dummies(X, drop_first=True)
    pca = PCA(n_components=n)
    return pca.fit_transform(X)

# rfe

def rfe_feature(X,Y,n):

    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor


    models=[LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(
    n_estimators=50)]

    names=["Linear", "Decision", "Random"]

    output=[]


    for model,name in zip(models, names):

        selector = RFE(estimator=model, n_features_to_select=n, step=10)

        data = selector.fit_transform(X, Y)

        output.append(
            (name, data)
        )

    return output
    
# =========================
# EVALUATION
# =========================

def evaluate(model, X_test, Y_test):

    pred = model.predict(X_test)

    return r2_score(Y_test, pred)


def train_model(
        model,
        params,
        X_train,
        Y_train,
        X_test,
        Y_test
):

    search = RandomizedSearchCV(
        model,
        params,
        n_iter=10,
        cv=3,
        scoring="r2",
        random_state=42,
        n_jobs=-1
    )

    search.fit(
        X_train,
        Y_train
    )

    return evaluate(
        search.best_estimator_,
        X_test,
        Y_test
    )


# =========================
# MODELS
# =========================

def run_models(
        X_train,
        Y_train,
        X_test,
        Y_test
):

    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor


    scores = {

        "Linear":
        train_model(
            Ridge(),
            {
                "alpha":
                uniform(0, 10)
            },
            X_train,
            Y_train,
            X_test,
            Y_test
        ),

        "SVMl":
        train_model(
            SVR(),
            {
                "C":
                np.logspace(-2, 2, 5),

                "epsilon":
                [0.01, 0.1],

                "kernel":
                ["linear"]
            },
            X_train,
            Y_train,
            X_test,
            Y_test
        ),

        "SVMnl":
        train_model(
            SVR(),
            {
                "C":
                np.logspace(-2, 2, 5),

                "gamma":
                ["scale", 0.01],

                "kernel":
                ["rbf"]
            },
            X_train,
            Y_train,
            X_test,
            Y_test
        ),

        "Decision":
        train_model(DecisionTreeRegressor(random_state=42),
                    {"max_depth":[None, 5, 10, 20], "min_samples_split":[2, 5, 10], "min_samples_leaf":[1, 2, 4], "criterion":["squared_error",
                                                                                                                               "friedman_mse"]},
            X_train,
            Y_train,
            X_test,
            Y_test
        ),
        
        
        "Random":
        train_model(RandomForestRegressor(random_state=42),
        
            { "n_estimators":[100,200,300], "max_depth":[10,20,30,None], "min_samples_split":[2,5,10], "min_samples_leaf":[1,2,4],
             "max_features":["sqrt","log2",None],"bootstrap":[True]},
            
            X_train,
            Y_train,
            X_test,
            Y_test
            
        ),

        
        "GradientBoost":

        train_model(GradientBoostingRegressor(),
        
            {"n_estimators":[100,200], "learning_rate":[0.01, 0.05,0.1], "max_depth":[3,5]},
        
            X_train,
            Y_train,
            X_test,
            Y_test
        ),
        
    }
    
    return pd.DataFrame(
        [scores]
    )


# =========================
# MULTIPLE K VALUES
# =========================

def compare_k_values(
        indep_X,
        dep_Y,
        k_values
):

    all_results = []

    for k in k_values:

        kbest = selectkbest(
            indep_X,
            dep_Y,
            k
        )

        X_train, X_test, Y_train, Y_test = split_scalar(
            kbest,
            dep_Y
        )

        result = run_models(
            X_train,
            Y_train,
            X_test,
            Y_test
        )

        best_model = result.idxmax(
            axis=1
        )[0]

        best_score = result.max(
            axis=1
        )[0]

        all_results.append({

            "K_Value": k,

            "Best_Model": best_model,

            "Best_Score": best_score
        })

    return pd.DataFrame(
        all_results
    )

def compare_pca_values(
        indep_X,
        dep_Y,
        pca_list
):

    result=[]

    for n in pca_list:

        pca_data = pca_features(
            indep_X,
            n
        )

        X_train,X_test,Y_train,Y_test=split_scalar(
            pca_data,
            dep_Y
        )

        score = run_models(
            X_train,
            Y_train,
            X_test,
            Y_test
        )

        best_model = score.idxmax(
            axis=1
        )[0]

        best_score = score.max(
            axis=1
        )[0]

        result.append({

            "PCA":n,

            "Best_Model":
            best_model,

            "Best_Score":
            best_score
        })

    return pd.DataFrame(
        result
    )  

def compare_rfe_values(
        indep_X,
        dep_Y,
        rfe_values
):

    final=[]

    for n in rfe_values:

        feature_sets = rfe_feature(
            indep_X,
            dep_Y,
            n
        )

        for name,data in feature_sets:

            X_train,X_test,Y_train,Y_test=split_scalar(
                data,
                dep_Y
            )

            score = run_models(
                X_train,
                Y_train,
                X_test,
                Y_test
            )

            best_score = score.max(
                axis=1
            )[0]

            final.append({

                "RFE":n,

                "Feature_Model":
                name,

                "Best_Score":
                best_score
            })

    return pd.DataFrame(
        final
    )