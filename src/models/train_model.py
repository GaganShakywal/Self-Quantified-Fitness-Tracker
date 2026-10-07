import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df=pd.read_pickle("../../data/interim/03_data_features.pkl")
df.head()
df.columns
# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df_train=df.drop(["set","participant","category"],axis=1)
X=df_train.drop("label",axis=1)
Y=df_train["label"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42,stratify=Y) # stratify to keep the same distribution of classes in train and test set

fig,ax=plt.subplots(figsize=(10,5))
df_train["label"].value_counts().plot(kind="bar",ax=ax,color="lightblue",label="Total")
Y_train.value_counts().plot(kind="bar",ax=ax,color="dodgerblue",label="Train")
Y_test.value_counts().plot(kind="bar",ax=ax,color="royalblue",label="Test")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
# splliting the features so that later we can evaluate the performance of different feature subsets which we add in the feature engineering step
basic_features=["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"]
square_features=["acc_r","gyr_r"]
pca_features=["pca_1","pca_2","pca_3"]
time_features=[col for col in X.columns if "_temp_" in col]
freq_features=[col for col in X.columns if ("_freq" in col) or ("_pse" in col)]
cluster_features=["cluster"]

print(f"Basic features: {len(basic_features)}")
print(f"Square features: {len(square_features)}")
print(f"PCA features: {len(pca_features)}")
print(f"Time features: {len(time_features)}")
print(f"Frequency features: {len(freq_features)}")
print(f"Cluster features: {len(cluster_features)}")

features_set_1=list(set(basic_features))
features_set_2=list(set(basic_features+square_features+pca_features))
features_set_3=list(set(features_set_2+time_features))
features_set_4=list(set(features_set_3+freq_features+cluster_features))
# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
# Now we will add features one by one and see how the performance improves by using a simple decision tree model and then all best features put together will be used for the final model training and selection
Learner=ClassificationAlgorithms()
max_features=10

selected_features, ordered_features, ordered_scores=Learner.forward_selection(
    X_train=X_train,
    y_train=Y_train,
    max_features=max_features,
)

selected_features=[
 'pca_1',
 'duration',
 'acc_z_freq_0.0_Hz_ws_14',
 'acc_y_temp_mean_ws_5',
 'gyr_x_freq_1.071_Hz_ws_14',
 'gyr_y_freq_2.143_Hz_ws_14',
 'acc_z_temp_std_ws_5',
 'gyr_x_freq_1.429_Hz_ws_14',
 'acc_z_temp_mean_ws_5',
 'gyr_r_freq_2.5_Hz_ws_14'
]

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

plt.figure(figsize=(10,5))
plt.plot(np.arange(1,max_features+1,1),ordered_scores)
plt.xlabel("Number Of featues")
plt.ylabel("accuracy")
plt.xticks(np.arange(1,max_features+1,1))
plt.show()

# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
# Grid search is the way to come up with the set of optimal hyperparameters
possible_features_set=[
    features_set_1,
    features_set_2,
    features_set_3,
    features_set_4,
    selected_features
]

features_names=[
    "Features Set 1",
    "Features Set 2",
    "Features Set 3",
    "Features Set 4",
    "Selected Features"
]
# iteration for setting the number of tiems we are running the model and then take the avg. of the accuracy

iterations =1
score_df=pd.DataFrame()

# Grid search code

for i, f in zip(range(len(possible_features_set)), features_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_features_set[i]]
    selected_test_X = X_test[possible_features_set[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = Learner.feedforward_neural_network(
            selected_train_X,
            Y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(Y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = Learner.random_forest(
            selected_train_X, Y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(Y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = Learner.k_nearest_neighbor(
        selected_train_X, Y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(Y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = Learner.decision_tree(
        selected_train_X, Y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(Y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = Learner.naive_bayes(selected_train_X, Y_train, selected_test_X)

    performance_test_nb = accuracy_score(Y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])
# it will score the accuracy of the test data after performing the trainig on train data with respective model
   
score_df.sort_values(by="accuracy",ascending=False,inplace=True)
plt.figure(figsize=(10,10))
sns.barplot(x="model",y="accuracy",hue="feature_set",data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7,1)
plt.legend(loc="lower right")
plt.show()
# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------
(
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
) = Learner.random_forest(
    X_train[features_set_4], Y_train, X_test[features_set_4], gridsearch=True
)
accuracy=accuracy_score(Y_test, class_test_y)
classes=class_test_prob_y.columns
# Now plot confusion matrix to see where the model is making mistakes
cm=confusion_matrix(Y_test,class_test_y,labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

'''Simple rule for 1️⃣ in this matrix:
If 1 is on the diagonal	If 1 is off the diagonal
✔️ Correct prediction	❌ Misclassification'''

# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------
participant_df=df.drop(["set","category"],axis=1)
X_train=participant_df[participant_df["participant"]!="A"].drop("label",axis=1)
Y_train=participant_df[participant_df["participant"]!="A"]["label"]

X_test=participant_df[participant_df["participant"]=="A"].drop("label",axis=1)
Y_test=participant_df[participant_df["participant"]=="A"]["label"]

X_train=X_train.drop("participant",axis=1)
X_test=X_test.drop("participant",axis=1)

fig,ax=plt.subplots(figsize=(10,5))
df_train["label"].value_counts().plot(kind="bar",ax=ax,color="lightblue",label="Total")
Y_train.value_counts().plot(kind="bar",ax=ax,color="dodgerblue",label="Train")
Y_test.value_counts().plot(kind="bar",ax=ax,color="royalblue",label="Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------
(
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
) = Learner.feedforward_neural_network(
    X_train[features_set_4], Y_train, X_test[features_set_4], gridsearch=True
)
accuracy=accuracy_score(Y_test, class_test_y)
classes=class_test_prob_y.columns
# Now plot confusion matrix to see where the model is making mistakes
cm=confusion_matrix(Y_test,class_test_y,labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------

# if you use feature_set_4 it will give you the same best accuracy irrespescive of the model used as included all the features

(
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
) = Learner.random_forest(
    X_train[selected_features], Y_train, X_test[selected_features], gridsearch=False
)
accuracy=accuracy_score(Y_test, class_test_y)
classes=class_test_prob_y.columns
# Now plot confusion matrix to see where the model is making mistakes
cm=confusion_matrix(Y_test,class_test_y,labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()