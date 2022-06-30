from random import Random
import sys
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from modelselection_conformal import ConformalRejectOptionGridSearchCV

from utils import *
from conformalprediction import ConformalPredictionClassifier, ConformalPredictionClassifierRejectOption, MyClassifierSklearnWrapper
from semifactual import SemifactualExplanation


def get_model(model_desc):
    if model_desc == "knn":
        return KNeighborsClassifier
    elif model_desc == "randomforest":
        return RandomForestClassifier
    elif model_desc == "gnb":
        return GaussianNB
    else:
        raise ValueError(f"Invalid value of 'model_desc' -- must be one of the following 'knn', 'dectree', 'gnb'; but not '{model_desc}'")

def get_model_parameters(model_desc):
    if model_desc == "knn":
        return knn_parameters
    elif model_desc == "randomforest":
        return random_forest_parameters
    elif model_desc == "gnb":
        return {}
    else:
        raise ValueError(f"Invalid value of 'model_desc' -- must be one of the following 'knn', 'dectree', 'gnb'; but not '{model_desc}'")


def compute_export_conformal_results(results_feasibility, results_sparsity, results_dist_orig, results_diversity):
    # Compute final statistics (TODO: Compute statsitics over all folds directly or per fold?)
    results_feasibility_mean, results_feasibility_var  = np.mean(results_feasibility), np.var(results_feasibility)
    results_sparsity_mean, results_sparsity_var = np.mean(results_sparsity), np.var(results_sparsity)
    results_dist_orig_mean, results_dist_orig_var = np.mean(results_dist_orig), np.var(results_dist_orig)
    results_diversity_mean, results_diversity_var = np.mean(results_diversity), np.var(results_diversity)

    # Export
    print(f"Feasibility: {results_feasibility_mean} \pm {results_feasibility_var}")
    print(f"Sparsity: {results_sparsity_mean} \pm {results_sparsity_var}")
    print(f"Dist to original sample: {results_dist_orig_mean} \pm {results_dist_orig_var}")
    print(f"Diversity: {results_diversity_mean} \pm {results_diversity_var}")


def compute_export_cnformal_perturbed_features_recovery_results(perturbed_features_recovery):
    # Compute final statistics (TODO: Compute statsitics over all folds directly or per fold?)
    perturbed_features_recovery_mean, perturbed_features_recovery_var = np.mean(perturbed_features_recovery), np.var(perturbed_features_recovery)

    # Export
    print(f"Perturbed features recovery: {perturbed_features_recovery_mean} \pm {perturbed_features_recovery_var}")



if __name__ == "__main__":
    #"""
    if len(sys.argv) != 3:
        print("Usage: <dataset> <model>")
        os._exit(1)

    # Specifications (provided as an input by the user)
    data_desc = sys.argv[1]
    model_desc = sys.argv[2]
    n_explanations = 3

    # Load data
    X, y = load_data(data_desc)
    print(X.shape)

    # Results/Statistics
    results_feasibility = [];results_feasibility_perturbed = []
    results_sparsity = [];results_sparsity_perturbed = []
    results_dist_orig = [];results_dist_orig_perturbed = []
    results_diversity = [];results_diversity_peturbed = []
    perturbed_features_recovery= []

    # In case of an extremly large majority class, perform simple downsampling
    if data_desc == "t21":
        rus = RandomUnderSampler()
        X, y = rus.fit_resample(X, y)

    # K-Fold
    for train_index, test_index in KFold(n_splits=n_folds, shuffle=True, random_state=None).split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # If necessary (in case of an highly imbalanced data set), apply Synthetic Minority Over-sampling Technique (SMOTE)
        if data_desc == "flip":
            sm = SMOTE(k_neighbors=1)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            X_test, y_test = sm.fit_resample(X_test, y_test)

        # Hyperparameter tuning
        model_search = ConformalRejectOptionGridSearchCV(model_class=get_model(model_desc), parameter_grid=get_model_parameters(model_desc), rejection_thresholds=reject_thresholds)
        best_params = model_search.fit(X_train, y_train)


        # Split training set into train and calibtration set (calibration set is needed for conformal prediction)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.2)

        # Fit & evaluate model and reject option
        model = get_model(model_desc)(**best_params["model_params"])
        model.fit(X_train, y_train)
        print(f"Model score: {model.score(X_train, y_train)}, {model.score(X_test, y_test)}")

        conformal_model = ConformalPredictionClassifier(MyClassifierSklearnWrapper(model))
        conformal_model.fit(X_calib, y_calib)
        print(f"Conformal predictor score: {conformal_model.score(X_train, y_train)}, {conformal_model.score(X_test, y_test)}")

        print(f'Rejection threshold: {best_params["rejection_threshold"]}')
        reject_option = ConformalPredictionClassifierRejectOption(conformal_model, threshold=best_params["rejection_threshold"])

        explanator = SemifactualExplanation(reject_option)

        # Select random subset of features which are going to be perturbed
        perturbed_features_idx = select_random_feature_subset(X_train.shape[1])
        print(f"Perturbed features: {perturbed_features_idx}")

        # For each sample in the test set, check if it is rejected
        y_rejects = []
        for i in range(X_test.shape[0]):
            x = X_test[i,:]
            if reject_option(x):
                y_rejects.append(i)
        print(f"{len(y_rejects)}/{X_test.shape[0]} are rejected")
        
        # Compute explanations for all rejected test samples
        results_feasibility_ = []
        for idx in y_rejects:
            try:
                x_orig = X_test[idx, :]

                X_sf = explanator.compute_diverse_explanations(x_orig, n_explanations=n_explanations)

                results_diversity.append(evaluate_diversity([np.abs(x_sf - x_orig) for x_sf in X_sf]))
                for x_sf in X_sf:
                    if reject_option(x_sf) and np.all(reject_option.criterion(x_sf) >= reject_option.criterion(x_orig)):
                        results_feasibility_.append(1. / (len(y_rejects) * n_explanations))
                    else:
                        results_feasibility_.append(0.)
                    results_sparsity.append(evaluate_sparsity(x_sf, x_orig))
                    results_dist_orig.append(evaluate_closeness(x_sf, x_orig))
            except Exception as ex:
                print(ex)
        results_feasibility.append(np.sum(results_feasibility_))

        # Find all samples in the test set that are rejected because of the perturbation
        X_test = apply_perturbation(X_test, perturbed_features_idx)  # Apply perturbation
        
        y_rejects_due_to_perturbations = []
        for i in range(X_test.shape[0]):    # Check which samples are now rejected
            x = X_test[i,:]
            if reject_option(x) and i not in y_rejects:
                y_rejects_due_to_perturbations.append(i)
        print(f"{len(y_rejects_due_to_perturbations)}/{X_test.shape[0]} are rejected due to perturbations")

        # Compute explanations for all rejected test samples
        results_feasibility_perturbed_ = []
        for idx in y_rejects_due_to_perturbations:
            try:
                x_orig = X_test[idx, :]

                X_sf = explanator.compute_diverse_explanations(x_orig, n_explanations=n_explanations)

                results_diversity_peturbed.append(evaluate_diversity([np.abs(x_sf - x_orig) for x_sf in X_sf]))
                for x_sf in X_sf:
                    if reject_option(x_sf) and np.all(reject_option.criterion(x_sf) >= reject_option.criterion(x_orig)):
                        results_feasibility_perturbed_.append(1. / (len(y_rejects_due_to_perturbations) * n_explanations))
                    else:
                        results_feasibility_perturbed_.append(0.)
                    results_sparsity_perturbed.append(evaluate_sparsity(x_sf, x_orig))
                    perturbed_features_recovery.append(evaluate_perturbed_features_recovery(x_sf, x_orig, perturbed_features_idx))
                    results_dist_orig_perturbed.append(evaluate_closeness(x_sf, x_orig))
            except Exception as ex:
                print(ex)
        results_feasibility_perturbed.append(np.sum(results_feasibility_perturbed_))

    # Compute and export final statistics
    print()
    print("****************** RESULTS ***************************")
    compute_export_conformal_results(results_feasibility, results_sparsity, results_dist_orig, results_diversity)
    
    print()
    print("********* Results for Perturbed features *************")
    compute_export_conformal_results(results_feasibility_perturbed, results_sparsity_perturbed, results_dist_orig_perturbed, results_diversity_peturbed)
    compute_export_cnformal_perturbed_features_recovery_results(perturbed_features_recovery)
