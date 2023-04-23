#importing packages
import numpy as numpyModule
import pandas as pandasModule
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from rulefit import RuleFit

#graphviz
import graphviz
from sklearn import tree




#loading the dataset
dataset=pandasModule.read_csv("OvarianCancer.csv")
print(dataset)

#Extracting the relavant features
relavantFeaturesFromKaggale=['HE4', 'NEU', 'CA125', 'ALB', 'Age', 'GLO', 'IBIL', 'HGB', 'PLT', 'Menopause']
#HE4 Human E Virus
#NEU Neutrophyll
#CA125 Cancer Antigen 125
#ALB Albumin
#Age 
#GLO Globulin
#IBIL Indirect Bilurubin
#HGB Haemoglobin
#PLT Platelet

#Extracting suitable features
purifiedData=dataset[relavantFeaturesFromKaggale]

#Preparing dataset for training and testing
X=purifiedData

Y=dataset["TYPE"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#classifier function

classifier=DecisionTreeClassifier()
#using fit for inductive learning
classifier.fit(X_train,Y_train)
Y_Pred=classifier.predict(X_test)
#the accuracy for inductive learning can be given below
accuracy=accuracy_score(Y_test,Y_Pred)*100
confusion_matrix_inductive=confusion_matrix(Y_test,Y_Pred)

#classifier function for deductive learning here we are making use of rule fit

classifier_deductive=RuleFit()
#splitting training and testing
classifier_deductive.fit(X_train.values,Y_train.values)
Y_pred_deductive=classifier_deductive.predict(X_test.values)
#calucating accuracy for deductive learning
accuracy_deductive=accuracy_score(Y_test,Y_pred_deductive.round())*100
confusion_matrix_deductive = confusion_matrix(Y_test, Y_pred_deductive.round())

#Graph for inductive and deductive learning can be given as follows

labels=['Negative','Positive']

x_axis=numpyModule.arange(len(labels))
width=0.35

fig,(axis1,axis2)=plot.subplots(1,2,figsize=(12,5))
#defining inductive learning graph
axis1.bar(x_axis-width/2,confusion_matrix_inductive[:,0],width,label="Predicted Benign Ovarian Cancer")
axis1.bar(x_axis+width/2,confusion_matrix_inductive[:,1],width,label="Predicted Ovarian Cancer")
axis1.set_ylabel('Counts')
axis1.set_title("Confusion matrix(Inductive Learning)")
axis1.set_xticks(x_axis)
axis1.set_xticklabels(labels)
axis1.legend()
#defining deductive learning learning graph
axis2.bar(x_axis-width/2,confusion_matrix_deductive[:,0],width,label="Predicted Benign Ovarian Cancer")
axis2.bar(x_axis+width/2,confusion_matrix_deductive[:,1],width,label="Predicted Ovarian Cancer")
axis2.set_ylabel("Counts")
axis2.set_title("Confusion matrix(Deductive Learning)")
axis2.set_xticks(x_axis)
axis2.set_xticklabels(labels)
axis2.legend()

fig.tight_layout()
plot.show()

print("Inductive learning model")
print(accuracy)

print("\n Deductive Learning model")
print(accuracy_deductive)


#for visulization of decision tree and Rulefit(Regression)
figure,axis=plot.subplots(figsize=(30,30))
tree.plot_tree(classifier,filled=True,fontsize=15,feature_names=X.columns)
#save the figure
plot.savefig('decision_tree_image.png')
plot.show()

#For rule fit regression
feature_importances_to_be_provided=classifier.feature_importances_
#Creating dataframe with feature names and their importances
importance_df=pandasModule.DataFrame(
    {"feature":X.columns,"importance":feature_importances_to_be_provided}
)
#Sorting the dataframe based on the importance
importance_df=importance_df.sort_values("importance",ascending=False)
#Plotting feature importances
figure,axis=plot.subplots(figsize=(30,30))
importance_df.plot(kind="bar",x="feature",y="importance",ax=axis,legend=None)
plot.ylabel("Importance")
plot.title("Feature importances for RuleFit model")
plot.savefig("regression_model_feature_importances.png")
plot.show()

