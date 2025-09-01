from sklearn.metrics import make_scorer, f1_score, accuracy_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import cross_validate

def cross_validation(classifier, X, Y):

    if X is None or Y is None:
        raise ValueError("Data not loaded. Please run load_data() first.")
    
    cv = 5
    scorers = {
        'accuracy_score': make_scorer(accuracy_score),
        'f1_score': make_scorer(f1_score, average='weighted'),
        'G_mean_score': make_scorer(geometric_mean_score)
    }
    cv_results = cross_validate(classifier, X, Y, cv=cv, scoring=scorers, return_train_score = True)

    # print("Accuracy score:", cv_results['test_accuracy_score'].mean())
    # print("Geometric mean score:", cv_results['test_G_mean_score'].mean())
    # print("F1 score:", cv_results['test_f1_score'].mean())
    # print("Fit time score:", cv_results['fit_time'].mean())

    acc = cv_results['test_accuracy_score'].mean()
    gmean = cv_results['test_G_mean_score'].mean()
    f1 = cv_results['test_f1_score'].mean()
    fit_time = cv_results['fit_time'].mean()
    
    return acc, gmean, f1, fit_time