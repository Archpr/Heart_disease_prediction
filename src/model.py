import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

def train_baseline_model(X_train, y_train):
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def tune_model(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': uniform(0.01, 0.1),
        'num_leaves': [20, 30, 40, 50],
        'max_depth': [-1, 10, 15, 20],
        'min_child_samples': [20, 30, 40, 50],
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
    }

    model = lgb.LGBMClassifier(random_state=42)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_
