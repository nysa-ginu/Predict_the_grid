import pandas as pd
import numpy as np
from readying_modelling_data import get_train_test_set
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

'''
Purpose of this module is to look for the best ML model that predicts the drivers that will be on the podium.
'''

def getting_final_df(prediction_df, round):

    '''
    This function takes the final predicted final Position updates the driver code and round and stores in csv

    Input:
        prediction_df : Dataframe with the actual label, predicted label, and probabilities of each class.
        round : round of the season

    Ouput:
    stores csv file in local machine.
    '''

    # Find the index of the standardized value in the standardized data
    index = np.where(X_test == round)[0][0]

    # Retrieve the value at the same index in the original dataframe
    original_round = val_set.iloc[index]['round']

    #retriving the driverCode
    prediction_df['driverCode'] = val_set[val_set['round'] == original_round]['driverCode'].reset_index(drop=True)
    prediction_df['round'] = original_round

    #Re-arranging columns
    col_order = ['round', 'driverCode', 'actual', 'predicted', 'proba_1', 'proba_2', 'proba_3']
    prediction_df = prediction_df[col_order]
    csv_name = "prediction_data/Season_2022_round_" + str(original_round) + ".csv"
    prediction_df.to_csv(csv_name, index=False)



def split_n_mapping_predictions(set_data):

    '''
    This fucntions splits and normalize the data and maps the final positon into 1, 2, 3, for the respective podium places
    and 0 otherwise.
    '''

    X = set_data.drop(['driverCode', 'finalPosition'], axis=1)
    X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

    y = set_data['finalPosition'].map(lambda x: x if x<=3 else 0)

    return X, y


def prediction_classification(model, type_work):

    '''
    This model predicts the labels and gives out the score.
    '''

    score = 0
    for circuit in X_test['round'].unique():

        X_test_circuit = X_test[X_test['round']==circuit]
        indices_X_test = X_test['round'].values == circuit
        y_test_circuit = y_test.loc[indices_X_test]

        # make predictions
        prediction_df = pd.DataFrame(model.predict_proba(X_test_circuit), columns = ['proba_0', 'proba_1', 'proba_2', 'proba_3'])
        prediction_df['actual'] = y_test_circuit.reset_index(drop = True)
        prediction_df['predicted'] = 0
        class_indices = ['proba_1', 'proba_2', 'proba_3']
        exclude_index = set()

        for classes in class_indices:
            class_prob_values = prediction_df[classes].values
            # Exclude indices from my_list
            updated_list = [value if index not in exclude_index else float('-inf') for index, value in enumerate(class_prob_values)]

            # Find the index of the highest value in the updated list
            max_index = max(enumerate(updated_list), key=lambda x: x[1])[0]
            exclude_index.add(max_index)
            class_name = int(classes.split("_")[1])
            prediction_df.loc[max_index, 'predicted'] = class_name

        if type_work == 'testing':
            getting_final_df(prediction_df, circuit)

        score += precision_score(prediction_df.actual, prediction_df.predicted, average='weighted')

    model_score = score / df[df.year == 2022]['round'].unique().max()
    return model_score


def train_logistic_regression():

    '''
    Trains the logistic regression model
    '''

    params={'penalty': ['l1', 'l2'],
            'solver': ['saga', 'liblinear'],
            'C': np.logspace(-3,1,20)}

    print("Starting LR")

    for penalty in params['penalty']:
        for solver in params['solver']:
            for c in params['C']:
                model_params = (penalty, solver, c)
                model = LogisticRegression(penalty = penalty, solver = solver, C = c, max_iter = 10000)
                model.fit(X_train, y_train)
                
                model_score = prediction_classification(model, "training")
                
                comparison_dict['model'].append('logistic_regression')
                comparison_dict['params'].append(model_params)
                comparison_dict['score'].append(model_score)
    print("LR Done!!!!!")


def train_random_forest():

    '''
    Trains the Random Forest classification model
    '''

    params={'criterion': ['gini', 'entropy'],
            'max_features': [0.8, 'auto', None]
            }

    print("Starting RFC")

    for criterion in params['criterion']:
        for max_features in params['max_features']:
            model_params = (criterion, max_features)
            model = RandomForestClassifier(criterion = criterion, max_features = max_features)
            model.fit(X_train, y_train)
            
            model_score = prediction_classification(model, "training")
            
            comparison_dict['model'].append('random_forest_classifier')
            comparison_dict['params'].append(model_params)
            comparison_dict['score'].append(model_score)

    print("RFC Done!!!!!")


def train_xgboost():

    '''
    Trains the XGBoost Classification model
    '''

    params={'booster' : ['gbtree', 'gblinear', 'dart'],
            'eta' : [0.3, 0.4, 0.5, 0.6, 0.7]
            }

    print("Starting XGB")

    for booster in params['booster']:
        for learning_rate in params['eta']:
            model_params = (booster, learning_rate)
            model = XGBClassifier(booster = booster, eta=learning_rate)
            model.fit(X_train, y_train)

            model_score = prediction_classification(model, "training")

            comparison_dict['model'].append('XGBoost_classifier')
            comparison_dict['params'].append(model_params)
            comparison_dict['score'].append(model_score)

    print("XGB Done!!!!!")


scaler = StandardScaler()
df, train_set, val_set = get_train_test_set()

X_train, y_train = split_n_mapping_predictions(train_set)
X_test, y_test= split_n_mapping_predictions(val_set)
comparison_dict ={'model':[],
                  'params': [],
                  'score': []}

train_logistic_regression()
train_random_forest()
train_xgboost()

print(pd.DataFrame(comparison_dict).groupby('model')['score'].max())


model_comparison = pd.DataFrame(comparison_dict)
model_comparison.to_csv("model_comparison.csv", index=False)

# getting the highest performing model
max_value = model_comparison['score'].max()
selected_row = model_comparison[model_comparison['score'] == max_value]

row_values = selected_row.values[0]
model, params = row_values[0], row_values[1]
if model == "logistic_regression":
    penalty, solver, c = params[0], params[1], params[2]
    model = LogisticRegression(penalty = penalty, solver = solver, C = c, max_iter = 10000)
    model.fit(X_train, y_train)
    model_score = prediction_classification(model, "testing")

elif model == "random_forest_classifier":
    criterion, max_features = params[0], params[1]
    model = RandomForestClassifier(criterion = criterion, max_features = max_features)
    model.fit(X_train, y_train)
    model_score = prediction_classification(model, "testing")

elif model == "XGBoost_classifier":
    booster, learning_rate = params[0], params[1]
    model = XGBClassifier(booster = booster, eta=learning_rate)
    model.fit(X_train, y_train)

    model_score = prediction_classification(model, "testing")

print("DONE!!!")


'''
for betting odds for the season 2022:
    https://www.formula1.com/en/latest/tags.betting.4DAHW78TqpGtu00QvfHuzk.html#default
'''
