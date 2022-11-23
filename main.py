import numpy as np
import pandas as pd
import random
from collections import deque
from sklearn.model_selection import train_test_split

from sklearn import tree


dataframe =  pd.read_csv('ionosphere.data', sep=",")

# split the dataset to train and test
train, test = train_test_split(dataframe)
X_test = test.iloc[: , :-1]
y_test = test.iloc[:,-1].tolist()

df = train

# name columns of dataframe
column_names = []
for i in range(34):
    column_names.append("feature"+str(i))
class_name = 'structure'
column_names.append(class_name)
df.columns = column_names


def get_split(df,feature_names):

    # STEP 1: Calculate gini(D) of the original dataset
    def gini_impurity (value_counts):
        n = value_counts.sum()
        p_sum = 0
        for key in value_counts.keys():
            p_sum = p_sum  +  (value_counts[key] / n ) * (value_counts[key] / n ) 
        gini = 1 - p_sum
        return gini

    class_value_counts = df.iloc[:,-1:].value_counts()
    #print(class_value_counts)
    #print(f'Number of samples in each class is:\n{class_value_counts}')

    gini_class = gini_impurity(class_value_counts)
    #print(f'\nGini Impurity of the class is {gini_class:.3f}')

    # STEP 2: 
    # Calculating  gini impurity for each of the random chosen feature
    def gini_split_f(attribute_name):
        attribute_values = df[attribute_name].value_counts()
        gini_F = 0 
        for key in attribute_values.keys():
            df_k = df[class_name][df[attribute_name] == key].value_counts()
            n_k = attribute_values[key]
            n = df.shape[0]
            gini_F = gini_F + (( n_k / n) * gini_impurity(df_k))
        return gini_F

    gini_feature ={}
    for key in feature_names:
        gini_feature[key] = gini_split_f(key)
        #print(f'Gini for {key} is {gini_feature[key]:.3f}')

    # STEP 3: 
    # The feature that has the smallest gini impurity is selected
    min_value = min(gini_feature.values())
    #print('The minimum value of Gini Impurity : {0:.3} '.format(min_value))

    for name, value in gini_feature.items():
        if value == min_value:
            selected_feature = name
    #print('The selected feature is: ', selected_feature)

    # STEP 4:
    # Find the best split point in the selected feature
    def find_split_point(feature_name):
        best_gini = 0.5
        best_split_point = 0
    
        column = list(df[feature_name])
        column = sorted(column)

        for i in range(1,len(column) - 1):
            gLeft, bLeft = 0, 0
            gRight, bRight = 0, 0 
            totalLeft, totalRight = 0,0
            giniLeft,giniRight = 0,0 
            if column[i] != column[i-1]:
                split_point = (column[i] + column[i-1])/2
                
                #count g and b in left half
                for j in range(i):
                    if df.iat[j,34] == 'g':
                        gLeft += 1
                    else:
                        bLeft += 1
                totalLeft = gLeft + bLeft
                gLeft = gLeft/totalLeft
                bLeft = bLeft/totalLeft
                giniLeft = 1 - (gLeft**2 + bLeft**2)

                #count g and b in right half
                for j in range(i,len(column)-1)      :
                    if df.iat[j,34] == 'g':
                        gRight += 1
                    else:
                        bRight += 1
                totalRight = gRight + bRight
                gRight = gRight/totalRight
                bRight = bRight/totalRight
                giniRight = 1 - (gRight**2 + bRight**2)

                giniA = (totalLeft/len(df))*giniLeft + (totalRight/len(df))*giniRight

                best_gini = min(best_gini,giniA)
                if best_gini == giniA:
                    best_split_point = split_point
        return best_split_point

    split_point = find_split_point(selected_feature)
    # print("Split point is", split_point)
    return split_point, selected_feature


# build tree using BFS - this function returns the level order traversal of the tree
def buildTree(df,features,traversal):
    q = deque()
    q.append(df)
    while q:
        cur_df = q.popleft()
        split_point, feature = get_split(cur_df,features)
        if split_point == 0 or len(cur_df) <= 20:
            traversal.append([feature,None])
            continue
        else:
            traversal.append([feature,split_point])
            df1, df2 = [x for _, x in cur_df.groupby(cur_df[feature] < split_point)]
            q.append(df1)
            q.append(df2)
    return traversal

#build a forest of tree_nums trees 
def buildForest(df,tree_nums):
    tree_traversals = []
    for _ in range(tree_nums):
        #randomly select 6 features from 34 features
        features = set()
        while len(features) < 6:
            feature_ind = random.randrange(0,34)
            features.add(column_names[feature_ind])
        features = list(features)

        #build tree 
        traversal = buildTree(df,features,[])
        tree_traversals.append(traversal)
    return tree_traversals


#predict using 1 tree
def predict_tree(data,tree_traversal):
    tag = None
    i = 0
    cur_branch = tree_traversal[i]
    feature, point = cur_branch
    while i < len(tree_traversal):
        index = int(feature[-1])-1
        if point != None and data[index] < point:
            i = (i*2)+1
            if i < len(tree_traversal):
                cur_branch = tree_traversal[i]
                feature, point = cur_branch
                tag = "b"
        elif point != None and data[index] > point:
            i = (i*2)+2
            if i < len(tree_traversal):
                cur_branch = tree_traversal[i]
                feature, point = cur_branch
                tag = "g"
        else:
            break
    return tag 


# predict using the forest - aggregate the results from all trees
def predict_forest(tree_traversals):
    final_results = []
    for i, row in X_test.iterrows():
        row = list(row)
        store = {}
        for traversal in tree_traversals:
            res = predict_tree(row,traversal)
            store[res] = store.get(res,0) + 1
        majority = max(store.values())
        for structure in store:
            if store[structure] == majority:
                final_results.append(structure)
                break
    return final_results


def computeAccuracy(result,answer):
    count = 0
    for i in range(len(result)):
        if result[i] == answer[i]:
            count += 1
    return round((count/len(result))*100)

'''
USE MODEL TO PREDICT 
'''

tree_traversals = buildForest(df,21)
final = predict_forest(tree_traversals)
accuracy = computeAccuracy(final,y_test)
print("The accuracy is:",accuracy,"%")
