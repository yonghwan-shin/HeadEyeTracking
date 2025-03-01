# %%
import numpy as np
from sklearn.pipeline import Pipeline
from FileHandling import *
from AnalysisFunctions import *
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.dists_kernels import FlatDist, ScipyDist
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, StratifiedKFold
from time import time
from sklearn import svm
import json
from sklearn.metrics import accuracy_score
import ast
from sklearn.metrics import RocCurveDisplay
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import joblib
import pickle

def getmax(data):
    return abs(data.max() - data.min())

def getstd(data):
    return data.std()

def getmean(data):
    return data.mean() - data[-1]

# def getdistance(data):
#     abs(data[-1] - data[0])
def getmaxvelocity(data):
    vels = abs(pd.Series(data).diff()[1:])
    return vels.max()

def getstdvelocity(data):
    vels = abs(pd.Series(data).diff()[1:])
    return vels.std()

def getmeanvelocity(data):
    vels = abs(pd.Series(data).diff()[1:])
    return vels.mean()

result_dir = Path.cwd() / "ML_results"
if not result_dir.exists():
    result_dir.mkdir()


def tune_with_halving_grid_search(x_train, y_train, param_grid, suffix):
    svc = svm.SVC(class_weight="balanced", random_state=42)

    start = time()
    halving_gs_results = HalvingGridSearchCV(
        svc, param_grid, cv=5, factor=3, min_resources="exhaust"
    ).fit(x_train, y_train)

    duration = time() - start

    results = pd.DataFrame(halving_gs_results.cv_results_)
    results.loc[:, "mean_test_score"] *= 100

    # take the most relevant columns and sort (for readability). Remember to sort on the iter columns first, so we see
    # the models with the most training data behind them first.
    results = results.loc[:, ("iter", "rank_test_score", "mean_test_score", "params")]
    results.sort_values(
        by=["iter", "rank_test_score"], ascending=[False, True], inplace=True
    )
    p = "halving_svc_results" + suffix + ".csv"
    results.to_csv(result_dir / p)
    return results, duration


def tune_with_grid_search(x_train, y_train, param_grid):
    svc = svm.SVC(kernel="rbf", class_weight="balanced", random_state=42)

    start = time()
    gs_results = GridSearchCV(svc, param_grid, cv=5).fit(x_train, y_train)
    duration = time() - start

    results = pd.DataFrame(gs_results.cv_results_)
    results.loc[:, "mean_test_score"] *= 100
    results.to_csv(result_dir / "svc_results.csv")

    # take the most relevant columns and sort (for readability)
    results = results.loc[:, ("rank_test_score", "mean_test_score", "params")]
    results.sort_values(by="rank_test_score", ascending=True, inplace=True)

    return results, duration
#%% ONNX TEST
model = joblib.load('ML_results/24_Walk_Head_23.pkl')
initial_type = [('float_input', FloatTensorType([None, 36]))]

# Convert the model
# onnx_model = convert_sklearn(model, initial_types=initial_type)
onnx_model = to_onnx(model, np.zeros(36))
# Save the ONNX model
with open('binary_classifier_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())


#%% TRAIN FOR ONLY HEAD

window = 24
cursor = 'Head'
repetitions=range(1,5)
posture = 'Walk'
subjects=range(24)
# subjects=range(1)
summary = pd.DataFrame(
        columns=[
            "subject",
            "posture",
            "cursor",
            "selection",
            "target",
            "repetition",
            "count",
            "H_data",
            "V_data",
            "Head_H_data",
            "Head_V_data",
            "Eye_H_data",
            "Eye_V_data",
            "type",
        ]
    )
# for subject, repetition in itertools.product(subjects, repetitions):
for subject in subjects:
    for repetition in repetitions:
        f, s = read_data(subject, repetition, cursor, "Dwell", posture)
        d = split_target(f)
        for t in range(9):
            trial_summary = {
                "subject": subject,
                "posture": posture,
                "cursor": cursor,
                "selection": "Dwell",
                "target": t,
                "repetition": repetition,
            }
            data = d[t]
            data.timestamp = data.timestamp - data.timestamp.values[0]
            data["success"] = data.cursor_angular_distance <= 3.0
            only_success = data[
                (data.success == True)
                | (data.target_name == "Target_" + str(t))
            ]
            if len(only_success) <= 0:
                continue
                # raise ValueError('no success frames', len(only_success))
            initial_contact_time = only_success.timestamp.values[0]
            before = data[data.timestamp <= initial_contact_time]
            dwell = data[data.timestamp > initial_contact_time]
            before_frame_count = math.floor((len(before) - 12) / window)
            after_frame_count = math.floor(len(dwell) / window)
            
            for i in range(len(dwell)-window+1):
                after_temp_data = dwell[i:i+window].copy()

                trial_summary = {
                    "subject": subject,
                    "posture": posture,
                    "cursor": cursor,
                    "selection": "Dwell",
                    "target": t,
                    "repetition": repetition,
                }
                trial_summary["count"] = i
                # trial_summary['data'] = row_to_add
                trial_summary["H_data"] = after_temp_data[
                    "horizontal_offset"
                ].to_numpy()
                trial_summary["V_data"] = after_temp_data[
                    "vertical_offset"
                ].to_numpy()
                trial_summary["Head_H_data"] = after_temp_data[
                    "head_horizontal_offset"
                ].to_numpy()
                trial_summary["Head_V_data"] = after_temp_data[
                    "head_vertical_offset"
                ].to_numpy()
                trial_summary["Eye_H_data"] = after_temp_data[
                    "eyeRay_horizontal_offset"
                ].to_numpy()
                trial_summary["Eye_V_data"] = after_temp_data[
                    "eyeRay_vertical_offset"
                ].to_numpy()

                trial_summary["type"] = "after"
                summary.loc[len(summary)] = trial_summary

            for i in range(len(before)-window+1):
                # if i == 0:
                #     before_temp_data = before[-window * (i + 1) :].copy()
                # else:
                #     before_temp_data = before[
                #         -window * (i + 1) : -window * i
                #     ].copy()
                # row_to_add = np.array(
                #     [before_temp_data['horizontal_offset'].to_numpy(),
                #      before_temp_data['vertical_offset'].to_numpy()])
                before_temp_data = before[i:i+window].copy()
                trial_summary = {
                    "subject": subject,
                    "posture": posture,
                    "cursor": cursor,
                    "selection": "Dwell",
                    "target": t,
                    "repetition": repetition,
                }
                trial_summary["count"] = i
                trial_summary["H_data"] = before_temp_data[
                    "horizontal_offset"
                ].to_numpy()
                trial_summary["V_data"] = before_temp_data[
                    "vertical_offset"
                ].to_numpy()
                trial_summary["Head_H_data"] = before_temp_data[
                    "head_horizontal_offset"
                ].to_numpy()
                trial_summary["Head_V_data"] = before_temp_data[
                    "head_vertical_offset"
                ].to_numpy()
                trial_summary["Eye_H_data"] = before_temp_data[
                    "eyeRay_horizontal_offset"
                ].to_numpy()
                trial_summary["Eye_V_data"] = before_temp_data[
                    "eyeRay_vertical_offset"
                ].to_numpy()
                trial_summary["type"] = "before"
                summary.loc[len(summary)] = trial_summary

summary.to_pickle("ML_dataset_onnx" + str(window) + ".pkl")

        # before_frame_count = math.floor((len(before) - 12) / window)
        # after_frame_count = math.floor(len(dwell) / window)

#%%
window = 24
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


summary = pd.read_pickle("ML_dataset_onnx" + str(window) + ".pkl")
labels = summary["type"].to_numpy()
mapping = {"after": 1, "before": 0}
labels = np.vectorize(mapping.get)(labels)
summary['distance'] = ((summary.H_data - summary.Eye_H_data)**2+(summary.V_data - summary.Eye_V_data)**2) **(1/2)
summary['distance' + "_max"] = summary['distance'].apply(getmax)
summary['distance' + "_std"] = summary['distance'].apply(getstd)
summary['distance' + "_mean"] = summary['distance'].apply(getmean)
summary["label"] = labels
for col in ["", "Head_", "Eye_"]:
    for dir in ["H", "V"]:
        colname = col + dir + "_data"
        d = summary[colname]
        summary[colname + "_max"] = d.apply(getmax)
        summary[colname + "_std"] = d.apply(getstd)
        summary[colname + "_mean"] = d.apply(getmean)
        # summary[colname+"_distance"] = d.apply(getdistance)

        summary[colname + "_max_velocity"] = d.apply(getmaxvelocity)
        summary[colname + "_std_velocity"] = d.apply(getstdvelocity)
        summary[colname + "_mean_velocity"] = d.apply(getmeanvelocity)


print("Data Ready")
sequences = summary.drop(columns= ['subject','posture','cursor','selection','target','repetition','count','H_data','V_data','Head_H_data','Head_V_data','Eye_H_data','Eye_V_data','type','distance'])
sequences.to_pickle("ML_sequence_onnx"+str(window)+".pkl")
#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import tf2onnx
import onnx
window = 24
sequences = pd.read_pickle("ML_sequence_onnx"+str(window)+".pkl")
sequences = sequences.dropna() 
num_sequences = sequences.shape[0]
num_features=sequences.shape[1]
# X = np.asarray(sequences.drop(columns=['label']).values).astype('float32')
# y = np.asarray(sequences['label'].values).astype('float32')
# X= np.array(sequences.drop(columns=['label']),dtype='object')
# y= np.array(list(sequences['label']),dtype='object')
# sequences_flattened = X.flatten()
# X = sequences.drop(columns=['label']).values.astype(np.float32)
def flatten_array_elements(row):
    return pd.Series(np.concatenate(row.values))

# X_flattened = sequences.drop(columns=['label']).flatten()
# apply(flatten_array_elements, axis=1)
X= sequences.drop(columns=['label'])
# y = sequences['label'].values.astype(np.float32)
y = sequences['label'].values.astype(np.float32)

# Check the data types
# print(f"X dtype: {X.dtype}, y dtype: {y.dtype}")


# Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42)
# X_train =X_train.astype(float)
# y_train = tf.cast(y_train , dtype=tf.float32)
# Define the neural network model
model = Sequential([
    Flatten(input_shape = (X.shape[1],)),
    # Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    # Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(16, activation = 'relu'),

    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(X, y)
print(f'Test Accuracy: {accuracy*100:.2f}%')
print(model.summary())
# Convert the TensorFlow model to ONNX
spec = (tf.TensorSpec((None, X.shape[1]), tf.float32, name="input"),)
output_path = "model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())

#%%
import shap
# explainer = shap.KernelExplainer(model.predict,X[:10],link="logit")
explainer = shap.Explainer(model.predict,X[:10])
shap_values=  explainer(X[:10])
# shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], link="logit")

# shap.summary_plot(shap_values, X[100:120],plot_type='bar')
# shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
# shap.plots.waterfall(shap_values[0])
shap.plots.bar(shap_values)
# shap.summary_plot(shap_values, X, plot_type="bar")



# %% MAKE DATASET SUMMARY FILE
# halving_results = pd.read_csv("ML_results/halving_svc_results.csv")
param_index = 0
hyperparameter = False
using_svm = True
make_file=False
# Divide into all conditions
result = pd.DataFrame(
    columns=[
        "window",
        "posture",
        "cursor",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "ROC",
    ]
)
# for window in [6]:
for window in [6, 10, 12, 15, 20, 24, 30, 36, 42, 48]:
    if make_file:  # make true for creating dataset for new window
        subjects = range(24)
        # subjects = [0]
        cursorTypes = ["Head", "Eye", "Hand"]
        repetitions = range(1, 5)
        postures = ["Stand", "Treadmill", "Walk"]
        summary = pd.DataFrame(
            columns=[
                "subject",
                "posture",
                "cursor",
                "selection",
                "target",
                "repetition",
                "count",
                "H_data",
                "V_data",
                "Head_H_data",
                "Head_V_data",
                "Eye_H_data",
                "Eye_V_data",
                "type",
            ]
        )

        for cursor, posture in itertools.product(cursorTypes, postures):

            for subject, repetition in itertools.product(subjects, repetitions):
                f, s = read_data(subject, repetition, cursor, "Dwell", posture)
                d = split_target(f)
                for t in range(9):
                    trial_summary = {
                        "subject": subject,
                        "posture": posture,
                        "cursor": cursor,
                        "selection": "Dwell",
                        "target": t,
                        "repetition": repetition,
                    }
                    data = d[t]
                    data.timestamp = data.timestamp - data.timestamp.values[0]
                    data["success"] = data.cursor_angular_distance <= 3.0
                    only_success = data[
                        (data.success == True)
                        | (data.target_name == "Target_" + str(t))
                    ]
                    if len(only_success) <= 0:
                        continue
                        # raise ValueError('no success frames', len(only_success))
                    initial_contact_time = only_success.timestamp.values[0]
                    before = data[data.timestamp <= initial_contact_time]
                    dwell = data[data.timestamp > initial_contact_time]
                    before_frame_count = math.floor((len(before) - 12) / window)
                    after_frame_count = math.floor(len(dwell) / window)

                    for i in range(after_frame_count):  # after contact
                        if i == 0:
                            after_temp_data = dwell[-window * (i + 1) :].copy()
                        else:
                            after_temp_data = dwell[
                                -window * (i + 1) : -window * i
                            ].copy()
                        # row_to_add = np.array(
                        #     [after_temp_data['horizontal_offset'].to_numpy(),
                        #      after_temp_data['vertical_offset'].to_numpy()])

                        trial_summary = {
                            "subject": subject,
                            "posture": posture,
                            "cursor": cursor,
                            "selection": "Dwell",
                            "target": t,
                            "repetition": repetition,
                        }
                        trial_summary["count"] = i
                        # trial_summary['data'] = row_to_add
                        trial_summary["H_data"] = after_temp_data[
                            "horizontal_offset"
                        ].to_numpy()
                        trial_summary["V_data"] = after_temp_data[
                            "vertical_offset"
                        ].to_numpy()
                        trial_summary["Head_H_data"] = after_temp_data[
                            "head_horizontal_offset"
                        ].to_numpy()
                        trial_summary["Head_V_data"] = after_temp_data[
                            "head_vertical_offset"
                        ].to_numpy()
                        trial_summary["Eye_H_data"] = after_temp_data[
                            "eyeRay_horizontal_offset"
                        ].to_numpy()
                        trial_summary["Eye_V_data"] = after_temp_data[
                            "eyeRay_vertical_offset"
                        ].to_numpy()

                        trial_summary["type"] = "after"
                        summary.loc[len(summary)] = trial_summary

                    for i in range(before_frame_count):  # before contact
                        if i == 0:
                            before_temp_data = before[-window * (i + 1) :].copy()
                        else:
                            before_temp_data = before[
                                -window * (i + 1) : -window * i
                            ].copy()
                        # row_to_add = np.array(
                        #     [before_temp_data['horizontal_offset'].to_numpy(),
                        #      before_temp_data['vertical_offset'].to_numpy()])
                        trial_summary = {
                            "subject": subject,
                            "posture": posture,
                            "cursor": cursor,
                            "selection": "Dwell",
                            "target": t,
                            "repetition": repetition,
                        }
                        trial_summary["count"] = i
                        trial_summary["H_data"] = before_temp_data[
                            "horizontal_offset"
                        ].to_numpy()
                        trial_summary["V_data"] = before_temp_data[
                            "vertical_offset"
                        ].to_numpy()
                        trial_summary["Head_H_data"] = before_temp_data[
                            "head_horizontal_offset"
                        ].to_numpy()
                        trial_summary["Head_V_data"] = before_temp_data[
                            "head_vertical_offset"
                        ].to_numpy()
                        trial_summary["Eye_H_data"] = before_temp_data[
                            "eyeRay_horizontal_offset"
                        ].to_numpy()
                        trial_summary["Eye_V_data"] = before_temp_data[
                            "eyeRay_vertical_offset"
                        ].to_numpy()
                        trial_summary["type"] = "before"
                        summary.loc[len(summary)] = trial_summary

        summary.to_pickle("ML_dataset" + str(window) + ".pkl")

    data = pd.read_pickle("ML_dataset" + str(window) + ".pkl")

    for condition, dataframe in data.groupby([data.posture, data.cursor]):
        start = time()
        
        """
        # print(condition)
        # H = np.zeros([1, window])
        # for i in summary['H_data']:
        #     H = np.dstack((H, i))
        # H = H[:, :, 1:]
        # H = np.swapaxes(H.T, 1, 2)

        # V = np.zeros([1, window])
        # for i in summary['V_data']:
        #     V = np.dstack((V, i))
        # V = V[:, :, 1:]
        # V = np.swapaxes(V.T, 1, 2)

        # HH = np.zeros([1, window])
        # for i in summary['Head_H_data']:
        #     HH = np.dstack((HH, i))
        # HH = HH[:, :, 1:]
        # HH = np.swapaxes(HH.T, 1, 2)
        
        # HV = np.zeros([1, window])
        # for i in summary['Head_V_data']:
        #     HV = np.dstack((HV, i))
        # HV = HV[:, :, 1:]
        # HV = np.swapaxes(HV.T, 1, 2)

        # EH = np.zeros([1, window])
        # for i in summary['Eye_H_data']:
        #     EH = np.dstack((EH, i))
        # EH = EH[:, :, 1:]
        # EH = np.swapaxes(EH.T, 1, 2)
        
        # EV = np.zeros([1, window])
        # for i in summary['Eye_H_data']:
        #     EV = np.dstack((EV, i))
        # EV = EV[:, :, 1:]
        # EV = np.swapaxes(EV.T, 1, 2)

        # DATA = np.hstack((H, V,HH,HV,EH,EV))
        """
        
        summary = dataframe.copy()
        labels = summary["type"].to_numpy()
        mapping = {"after": 1, "before": 0}
        labels = np.vectorize(mapping.get)(labels)

        for col in ["", "Head_", "Eye_"]:
            for dir in ["H", "V"]:
                colname = col + dir + "_data"
                d = summary[colname]
                summary[colname + "_max"] = d.apply(getmax)
                summary[colname + "_std"] = d.apply(getstd)
                summary[colname + "_mean"] = d.apply(getmean)
                # summary[colname+"_distance"] = d.apply(getdistance)

                summary[colname + "_max_velocity"] = d.apply(getmaxvelocity)
                summary[colname + "_std_velocity"] = d.apply(getstdvelocity)
                summary[colname + "_mean_velocity"] = d.apply(getmeanvelocity)
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score

        summary["label"] = labels
        # Split the data into training and testing sets
        input_columns = [
            "H_data_max",
            "H_data_std",
            "H_data_mean",
            "H_data_max_velocity",
            "H_data_std_velocity",
            "H_data_mean_velocity",
            "V_data_max",
            "V_data_std",
            "V_data_mean",
            "V_data_max_velocity",
            "V_data_std_velocity",
            "V_data_mean_velocity",
            "Head_H_data_max",
            "Head_H_data_std",
            "Head_H_data_mean",
            "Head_H_data_max_velocity",
            "Head_H_data_std_velocity",
            "Head_H_data_mean_velocity",
            "Head_V_data_max",
            "Head_V_data_std",
            "Head_V_data_mean",
            "Head_V_data_max_velocity",
            "Head_V_data_std_velocity",
            "Head_V_data_mean_velocity",
            "Eye_H_data_max",
            "Eye_H_data_std",
            "Eye_H_data_mean",
            "Eye_H_data_max_velocity",
            "Eye_H_data_std_velocity",
            "Eye_H_data_mean_velocity",
            "Eye_V_data_max",
            "Eye_V_data_std",
            "Eye_V_data_mean",
            "Eye_V_data_max_velocity",
            "Eye_V_data_std_velocity",
            "Eye_V_data_mean_velocity",
        ]
        DATA = summary[input_columns]
        lables = summary["label"]

        # Leave-One-Out Cross-validation
        participants = summary["subject"].unique()
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        roc_aucs = []
        
        for participant in participants:
            
            train_df = summary[summary.subject != participant]
            test_df = summary[summary.subject == participant]
            X_train = train_df[input_columns]
            y_train = train_df["label"]
            X_test = test_df[input_columns]
            y_test = test_df["label"]
            print(X_train.shape[1])
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            filename = (
                    str(window)
                    + "_"
                    + str(condition[0])
                    + "_"
                    + str(condition[1])
                    + "_"
                    + str(participant)
                    + ".pkl"
                )
            try:
                clf = joblib.load("ML_results/" + filename)
                print(f'estimator loaded : {filename}')
            except:
                clf = SVC(
                    kernel="rbf", probability=True, class_weight="balanced", random_state=42
                )
                clf.fit(X_resampled, y_resampled)
                # joblib.dump(clf, "ML_results/" + filename)

            # Make predictions
            y_pred = clf.predict(X_test)

            # Evaluate the predictions
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="binary"
            )
            try:
                roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

                # Append metrics to arrays
                accuracies.append(acc)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                roc_aucs.append(roc_auc)

            except Exception as e:

                print("error",e.args)
        result.loc[len(result)] = {
            "window": window,
            "posture": condition[0],
            "cursor": condition[1],
            "accuracy": accuracies,
            "f1": f1_scores,
            "precision": precisions,
            "recall": recalls,
            "ROC": roc_aucs,
        }
        print(window, condition, math.ceil(time() - start), "sec")

        """
        X_train, X_test, y_train, y_test = train_test_split(DATA, labels, test_size=0.2, random_state=42)
        params = {
                    # 'C': [0.1, 0.5, 1, 2, 5, 10],
                    # 'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                    # 'tol': [1e-3, 1e-2]
                     'kernel':'rbf',
                    'class_weight':'balanced',
                    'probability':True,
                    'random_state':42
                }
        svc = svm.SVC(**params)

        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        print(condition,window)
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix, plot_roc_curve
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        plot_confusion_matrix(svc, X_test, y_test)
        plt.title('Confusion Matrix')
        plt.show()
        plot_roc_curve(svc, X_test, y_test)
        plt.title('ROC Curve')
        plt.show()
        from sklearn.inspection import permutation_importance
        result = permutation_importance(svc, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        # Get feature importances
        importance = result.importances_mean

        # Plot feature importances
        features = DATA.columns
        indices = np.argsort(importance)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importance[indices], align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Permutation Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance using Permutation Importance')
        plt.show()
        """
#%%
result = pd.read_pickle("ML_summaries.pkl")
result = result[result.window <=45]
result.window = result.window * 1000 /60
mean_of_list = lambda x: np.mean(x)
std_of_list = lambda x: np.std(x)

df_means = result[['accuracy','f1','precision','recall','ROC']].applymap(mean_of_list)
df_stds =result[['accuracy','f1','precision','recall','ROC']].applymap(std_of_list)
for column in df_means.columns:
    result[f'{column}_mean'] = df_means[column]
    result[f'{column}_std'] = df_stds[column]
    fig = px.line(result, x='window', y=f'{column}_mean', color='cursor', error_y=None,     facet_col='posture',markers=True,
                 labels={'mean': 'Mean Measurement', 'window': 'Window','cursor':'Modality'},
                 category_orders={'Mobility': ['Stand','Treadmill','Circuit'], "cursor": ['Eye','Head','Hand']},
                 title='Mean Measurements by Window, Posture, and Cursor',
                 template='plotly_white',
                 symbol='cursor')

    # Customize the layout
    if column == "ROC": column ="ROC AUC"
    fig.update_layout(
        title={
            'text':f'Mean {column} by Window, Posture, and Cursor',
               'x':0.5
               },
        xaxis_title='Window (ms)',
          annotations=[
        dict(text="Stand",),
        dict(text="Treadmill",),
        dict(text="Circuit",),
        
    ]
    )
    fig.update_xaxes(    
        title_text="Window (ms)",row=1,col=2
    )
    fig.update_xaxes(    
        title_text="Window (ms)",row=1,col=3
    )
    # if col == ""
    fig.update_layout(        
        yaxis_title='ROC AUC',
        height=350,width=800
    )
    fig.update_xaxes(ticks="outside", dtick=100)
    fig.update_yaxes(range=[0.75,1])
    fig.show()

    
    
# %%
# plt.show()
# y_score = svc.predict_proba(X_test.reshape(len(X_test)))
# if using_svm:
#     if hyperparameter == True:
#         # Initialize and train the SVM classifier
#         svm_classifier = SVC(
#             # kernel='rbf',
#             class_weight='balanced',
#             random_state=42)  # You can choose different kernels, e.g., 'linear', 'poly', 'rbf'

#         svc_params = {
#             'C': [0.1, 0.5, 1, 2, 5, 10],
#             'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
#             'tol': [1e-3, 1e-2]
#         }

#         # gs_results, gs_duration = tune_with_grid_search(X_train.reshape(len(X_train), 2 * window), y_train, svc_params)
#         halving_results, halving_duration = tune_with_halving_grid_search(
#             X_train.reshape(len(X_train), 2 * window),
#             y_train, svc_params,
#             suffix=str(window) + str(
#                 condition[0]) + str(condition[1]))
#         print(halving_results.head())

#         score = halving_results['mean_test_score'].iloc[0]
#         params = halving_results['params'].iloc[0]
#     else:
#         suffix = str(window) + str(condition[0]) + str(condition[1])
#         p = 'halving_svc_results' + suffix + '.csv'
#         hp = pd.read_csv(result_dir / p)
#         params = ast.literal_eval(hp.iloc[0].params)
#     svc = svm.SVC(**params)

#     svc.fit(X_train.reshape(len(X_train), 2 * window), y_train)
#     y_pred = svc.predict(X_test.reshape(len(X_test), 2 * window))
#     y_pred_roc = svc.decision_function(X_test.reshape(len(X_test), 2 * window))
#     accuracy = accuracy_score(y_test, y_pred)
#     # y_score = svc.predict_proba(X_test.reshape(len(X_test)))

#     # fig = plt.figure(figsize=(8, 8))
#     # fig.set_facecolor('white')
#     # ax = fig.add_subplot()
#     # # RocCurveDisplay.from_estimator(svc, X_train.reshape(len(X_train), 2 * window), y_train, ax=ax)
#     # RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_pred_roc)
#     # ax.plot([0, 1], [0, 1], color='red', label='Random Model')
#     # ax.legend()
#     # p = 'ROC_Curve' + suffix + '.png'
#     # plt.savefig(result_dir / p)
#     # # plt.show()

#     print(condition, "\t\tAccuracy:", accuracy)
#     condition_summary = {'posture': condition[0],
#                          'cursor': condition[1],
#                          'window': window,
#                          'accuracy': accuracy, }

#     conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

#     # Extract true positives, false positives, and false negatives from the confusion matrix
#     TP = conf_matrix[1, 1]  # True positive
#     FP = conf_matrix[0, 1]  # False positive
#     FN = conf_matrix[1, 0]  # False negative

#     # Calculate precision, recall, and F1 score
#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     f1 = 2 * (precision * recall) / (precision + recall)

#     # Alternatively, you can use the f1_score function from scikit-learn
#     # f1_sklearn = f1_score(y_test, y_pred)
#     try:
#         condition_summary['f1'] = f1
#         condition_summary['precision'] = precision
#         condition_summary['recall'] = recall
#     except Exception as e:
#         result.loc[len(result)] = condition_summary
#         print(e);
#         continue
#     result.loc[len(result)] = condition_summary
# else:  # using kneighbor
#     n_splits = 5
#     pipeline = GridSearchCV(
#         Pipeline([
#             # ('normalize', TimeSeriesScalerMinMax()),
#             ('knn', KNeighborsTimeSeriesClassifier())
#         ]),
#         {'knn__n_neighbors': [5, 25], 'knn__weights': ['uniform', 'distance']},
#         cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     )
#     # Plot our timeseries
#     colors = ['b', 'r']
#     fig, axs = plt.subplots(2, 1)
#     for ts, label in zip(X_train, y_train):
#         axs[0].plot(ts[0], c=colors[label], alpha=0.5)
#         axs[1].plot(ts[1], c=colors[label], alpha=0.5)
#         # print(ts,label)
#     axs[0].set_title('The timeseries in the dataset' + str(condition))
#     plt.tight_layout()
#     plt.show()

#     print('Performing hyper-parameter tuning of KNN classifier... ')
#     pipeline.fit(X_train, y_train)
#     results = pipeline.cv_results_
#     header_str = '|'
#     columns = ['n_neighbors', 'weights']
#     columns += ['score_fold_{}'.format(i + 1) for i in range(n_splits)]
#     for col in columns:
#         header_str += '{:^12}|'.format(col)
#     print(header_str)
#     print('-' * (len(columns) * 13))

#     for i in range(len(results['params'])):
#         s = '|'
#         s += '{:>12}|'.format(results['params'][i]['knn__n_neighbors'])
#         s += '{:>12}|'.format(results['params'][i]['knn__weights'])
#         for k in range(n_splits):
#             score = results['split{}_test_score'.format(k)][i]
#             score = np.around(score, 5)
#             s += '{:>12}|'.format(score)
#         print(s.strip())

#     best_comb = np.argmax(results['mean_test_score'])
#     best_params = results['params'][best_comb]

#     print()
#     print('Best parameter combination:')
#     print('weights={}, n_neighbors={}'.format(best_params['knn__weights'],
#                                               best_params['knn__n_neighbors']))
#     # eucl_dist = FlatDist(ScipyDist())
#     # clf = KNeighborsTimeSeriesClassifier(n_neighbors=3, distance=eucl_dist)
#     # clf.fit(X_train, y_train)
#     # print(clf.get_fitted_params())
#     # y_pred = clf.predict(X_test)
#     # # unique,counts = np.unique(y_pred,return_counts=True)
#     # print(accuracy_score(y_test, y_pred))
# %%plots
import seaborn as sns
from sklearn import metrics

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
# %%
import seaborn as sns

data = result.reset_index()
# col = 'accuracy'
# col = 'precision'
# col = 'recall'
col = "f1"
if col == "f1":
    yaxis_name = "f1 score"
elif col == "accuracy":
    yaxis_name = "accuracy"
elif col == "precision":
    yaxis_name = "precision"
elif col == "recall":
    yaxis_name = "recall"
g = sns.catplot(
    data=data,
    x="window",
    y=col,
    hue="cursor",
    col="posture",
    kind="bar",
    hue_order=["Head", "Eye", "Hand"],
)
# g = sns.catplot(data=data, x="window", y=col, hue="cursor",
#                 col="posture",
#                 kind='point', markers=True, marker='o')
# g.map(plt.axhline, y=0.8, color=".7", dashes=(2, 1), zorder=0)
g.map(plt.axhline, y=0.9, color=".7", dashes=(2, 1), zorder=0).set_axis_labels(
    "window size", yaxis_name
).set_titles("Modality: {col_name}").tight_layout(w_pad=0)
# plt.show()
plt.ylim(0.9, 1.0)
plt.savefig("ML_" + col + ".png")
