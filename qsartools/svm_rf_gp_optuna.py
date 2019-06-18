import argparse
import uuid
import pickle
import optuna
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('X', type=str, help='data for X npz format')
    parser.add_argument('Y', type=str, help='data for Y npz format')
    parser.add_argument('target', type=str, help='name of target, it will be database name')
    parser.add_argument('--testsize', type=float, help='testsize of train/test split default = 0.2', default=0.2)
    parser.add_argument('--n_trials', type=int, help='number of trials default = 200', default=200)
    parser.add_argument('--outpath', type=str, help='path for output default is lgb_output', default='svc_rf_gp_output')
    return parser

def objectives(trial):
    trial_uuid = str(uuid.uuid4())
    trial.set_user_attr('uuid', trial_uuid)
    classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForest', 'GP'])
    if classifier_name == 'SVC':
        svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
        svc_gamma = trial.suggest_loguniform('svc_gamma', 1e-10, 1e10)
        classifier_obj = SVC(C=svc_c, gamma=svc_gamma)
        trial_uuid += 'svc_'
    elif classifier_name == 'RandomForest':
        rf_max_depth = int(trial.suggest_loguniform('rf_max_depth', 2, 32))
        classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=10)
        trial_uuid += 'rf_'
    else:
        classifier_obj = GaussianProcessClassifier()
        trial_uuid += 'gp_'
    classifier_obj.fit(train_x, train_y)
    score = cross_val_score(classifier_obj, train_x, train_y, n_jobs=4, cv=5)
    val_accuracy = 1.0-score.mean()
    trial.set_user_attr('val_acc', val_accuracy)
    #classifier_obj.fit(train_x, train_y)

    y_pred_train = classifier_obj.predict(train_x)
    y_pred_test = classifier_obj.predict(test_x)

    acc_train = accuracy_score(train_y, y_pred_train)
    acc_test =  accuracy_score(test_y, y_pred_test)
    
    error_train = 1.0 - acc_train
    error_test = 1.0 - acc_test

    trial.set_user_attr('train_error', error_train)
    trial.set_user_attr('train_acc', acc_train)
    trial.set_user_attr('test_error', error_test)
    trial.set_user_attr('test_arcc', acc_test)

    if not os.path.exists('svc_rf_gp_output'):
        os.mkdir('svc_rf_gp_output')
    with open('svc_rf_gp_output/' + f'{trial_uuid}.pkl', 'wb') as fp:
        pickle.dump(classifier_obj, fp)
    return error_test

if __name__=='__main__':
    parser = make_parser()
    args = parser.parse_args()
    study = optuna.create_study(storage=f'sqlite:///{args.target}_svc_rf.db')
    X = np.load(args.X)['arr_0']
    Y = np.load(args.Y)['arr_0']
    print(X.shape)
    print(Y.shape)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=args.testsize)
    study.optimize(objectives, n_trials=args.n_trials)
    print(study.best_params)
    print(study.best_value)
    df = study.trials_dataframe()
    df.to_csv(f'{args.outpath}/optuna_svc_rf.csv')
