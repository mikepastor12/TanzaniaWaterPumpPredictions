##########################################################################################
#   Tanzania_main.py
#
#   Predict the health of water pumps using a Neural Network
#
#   The data for this competition comes from the Taarifa waterpoints dashboard,
#   which aggregates data from the Tanzania Ministry of Water.
#
#   Mike Pastor 10/19/2022
#
# https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/24/

# Local runtime variables
#
EPOCH_COUNT=6


TRAIN_VAL_SPLIT = 0.01      # validation set %


#########################################################################
#  Import the necessary libraries
#
import numpy as np
np.set_printoptions(precision=2)
from numpy import random
import pandas as pd
from datetime import datetime
import scipy.stats as stats

import tensorflow as tf
from keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid

from keras.regularizers import l2

#  %matplotlib widget
import matplotlib.pyplot as plt
# plt.style.use('./pennwick.mplstyle')

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


#################################################################
#  Track the overall time for training & submission preparation
#
global_now = datetime.now()
global_current_time = global_now.strftime("%H:%M:%S")
print("##############  Starting up...  - Current Time =", global_current_time)


# load TRAIN datasets ###########################################################
#
train_X = pd.read_csv('./TrainingSetValues.csv')
train_Y = pd.read_csv('./TrainingSetLABELS.csv')

# Merge the Y onto the X to be sure they are matched
#  This adds 'status_group' LABEL onto each X example row
#
train_X = train_X.merge( train_Y, how='inner', on='id' )

#########################################################
# replacing Y  string values with numerics
#    This is becomes part of our Tanzania Data Dictionary
#
train_X['status_group'].replace(\
    ['functional', 'functional needs repair', 'non functional'],\
    [0, 1, 2], inplace=True)

# Used the synchronized - quantitative  LABEL for Y
train_Y = pd.DataFrame( train_X['status_group'], columns=['status_group'] )


########################################################
#  Load the TEST dataset which is used for the Submission preparation
#
test_X = pd.read_csv('./TestSetValues.csv')
# test_df_sample.info()

print('train_X dataset dimensions are:', train_X.shape)
print('train_Y dataset dimensions are:', train_Y.shape)
print('Test X  dataset dimensions are:', test_X.shape)


#######################################################################
# drop variables with missing values >=75% in the train dataframe
#
i = 0
for col in train_X.columns:
    if (train_X[col].isnull().sum() / len(train_X[col]) * 100) >= 75:
        print("Dropping column", col)
        train_X.drop(labels=col, axis=1, inplace=True)
        # Also in test
        test_X.drop(labels=col, axis=1, inplace=True)
        i = i + 1

if i > 0:
    print("#####  Total number of NULL  columns dropped in train/test dataframes TRAIN-TEST == ", i,  train_X.shape, test_X.shape )


###############################################################  TBD
# define list  categorical variables (columns)
categorical = list(train_X.select_dtypes('object').columns)
print(f"Categorical variables (columns) are: {categorical}")
# define list   numerical variables (columns)
numerical = list(train_X.select_dtypes('number').columns)
print(f"Numerical variables (columns) are: {numerical}")


################################################################
#  Take a subset of the columns on both TRAIN and TEST X

#  Categorical variables (columns) are: ['date_recorded', 'funder', 'installer', 'wpt_name', /
#  'basin', 'subvillage', 'region', 'lga', 'ward', 'public_meeting', 'recorded_by', /
#  'scheme_management', 'scheme_name', 'permit', 'extraction_type', 'extraction_type_group', 'extraction_type_class', /
#  'management', 'management_group', 'payment', 'payment_type', 'water_quality', 'quality_group', /
#  'quantity', 'quantity_group', 'source', 'source_type', 'source_class', 'waterpoint_type', 'waterpoint_type_group', 'status_group']

#  Numerical variables (columns) are: ['id', 'amount_tsh', 'gps_height', 'longitude', 'latitude', \
#  'num_private', 'region_code', 'district_code', 'population', 'construction_year']

#################################
#  Only use the quantitative columns that have
#     at least a .05 correlation with our target - status_group
#
#  Deleted:  date_recorded',
#  'id',  'status_group',

train_X = train_X[ [   'payment', 'water_quality', 'source', 'source_type', 'source_class',\
                       'quantity', 'management',  'waterpoint_type', 'waterpoint_type_group', \
                       'basin', 'funder', 'installer', 'lga',\
                       'extraction_type', 'extraction_type_class', 'scheme_name', 'scheme_management',\
                       'construction_year', 'permit', 'public_meeting',\
                       'amount_tsh', 'gps_height', 'latitude',\
                       'region_code', 'district_code', 'population', 'num_private' ]  ]

# Save IDs for Submission file preparation below
test_X_Saved_IDs = test_X[ [  'id' ] ]
test_X = test_X[ [    'payment', 'water_quality', 'source', 'source_type', 'source_class',\
                       'quantity', 'management',  'waterpoint_type', 'waterpoint_type_group', \
                       'basin', 'funder', 'installer', 'lga',\
                       'extraction_type', 'extraction_type_class', 'scheme_name', 'scheme_management',\
                       'construction_year', 'permit', 'public_meeting',\
                       'amount_tsh', 'gps_height', 'latitude',\
                       'region_code', 'district_code', 'population', 'num_private' ]  ]



#######################################
#   Correlation matrix
#

#   form PANDA  correlation matrix
# matrix = train_X.corr()
# print("Correlation matrix is : \n")
# print(matrix)

#  Only use the quantitative columns that have at least a .05 correlaton with our target - status_group
#

#                     status_group        id  ...  population  construction_year
# status_group           1.000000  0.004049  ...   -0.017759          -0.043342
# id                     0.004049  1.000000  ...   -0.002813          -0.002082
# amount_tsh            -0.053702 -0.005321  ...    0.016288           0.067915
# gps_height            -0.114029 -0.004692  ...    0.135003           0.658727
# longitude              0.004366 -0.001348  ...    0.086590           0.396732
# latitude              -0.014547  0.001718  ...   -0.022152          -0.245278
# num_private           -0.005021 -0.002629  ...    0.003818           0.026056
# region_code            0.108640 -0.003028  ...    0.094088           0.031724
# district_code          0.065687 -0.003044  ...    0.061831           0.048315
# population            -0.017759 -0.002813  ...    1.000000           0.260910
# construction_year     -0.043342 -0.002082  ...    0.260910           1.000000


###################################################################
#   CHI Square test to find meaningful Categorical relationships
#
#
# # convert it back to strings for th chi test
# train_X['status_group'].replace(\
#     [0, 1, 2], ['functional', 'functional needs repair', 'non functional'], inplace=True)
#
# # create contingency table
# data_crosstab = pd.crosstab(train_X['status_group'],
#                             train_X['extraction_type'],
#                             margins=True, margins_name="Total")
# print( data_crosstab )
# # significance level
# alpha = 0.05
#
# # Calcualtion of Chisquare
# chi_square = 0
# rows = train_X['status_group'].unique()
# columns = train_X['extraction_type'].unique()
# for i in columns:
#     for j in rows:
#         O = data_crosstab[i][j]
#         E = data_crosstab[i]['Total'] * data_crosstab['Total'][j] / data_crosstab['Total']['Total']
#         chi_square += (O - E) ** 2 / E
#
# # The p-value approach
# print("Approach 1: The p-value approach to hypothesis testing in the decision rule")
# p_value = 1 - stats.chi2.cdf(chi_square, (len(rows) - 1) * (len(columns) - 1))
# conclusion = "Failed to reject the null hypothesis."
# if p_value <= alpha:
#     conclusion = "Null Hypothesis is rejected."
#
# print("chisquare-score is:", chi_square, " and p value is:", p_value)
# print(conclusion)
#
# # The critical value approach
# print("\n--------------------------------------------------------------------------------------")
# print("Approach 2: The critical value approach to hypothesis testing in the decision rule")
# critical_value = stats.chi2.ppf(1 - alpha, (len(rows) - 1) * (len(columns) - 1))
# conclusion = "Failed to reject the null hypothesis."
# if chi_square > critical_value:
#     conclusion = "Null Hypothesis is rejected."
#
# print("chisquare-score is:", chi_square, " and critical value is:", critical_value)
# print(conclusion)
#
# exit(-1)
#

##############################################################
#   Let's work on the train_X  CATEGORICAL columns
#       - Same for TEST

print( 'before get_dummies Train_X ==', train_X.shape )
print( 'before get_dummies test_X ==', test_X.shape )
train_X = pd.get_dummies( train_X )
test_X = pd.get_dummies( test_X )

# Align the number of features across test sets based on train dataset
train_X, test_X = train_X.align(test_X, join='left', axis=1, fill_value=0 )

print( 'AFTER  Train_X ==', train_X.shape )
print( 'AFTER test_X ==', test_X.shape )


#
#  Now split the TRAINING dataset into a VALIDATION set also
#
####################################################################################
#  TBD   Create a split Training and new Validation from the  merged TRAIN dataset
#
X_TRAIN_SplitDF, X_VALIDATE_SplitDF,  Y_TRAIN_SplitDF, Y_VALIDATE_SplitDF =\
    train_test_split( train_X,  train_Y, test_size=TRAIN_VAL_SPLIT )

print(f"***********  Train X  set is {len(X_TRAIN_SplitDF)} rows  and Validation set is {len(X_VALIDATE_SplitDF)} rows ")
print(f"***********  Train Y  set is {len(Y_TRAIN_SplitDF)} rows  and Validation set is {len(Y_VALIDATE_SplitDF)} rows ")


############################################################################################
# Setup the Layer structure for our NN with a Sequential model
# Andrew's example
#    32 / 16 / 8 / 4 / 12 / 10
#
#       Summary of steps
#
#       Get more Training examples  -> Fixes High Variance - Doesn't help Bias
#       Try smaller set of features - Simplify the model ->  Fixes High Variance
#       Try Additional features - Model needs more data  ->  Fixes High Bias
#       Try adding polynomial features (x squared, etc) -> Fixes High Bias
#       Decrease Lambda ->  Fixes High Bias
#       Increase Lambda ->   Fixes High Variance

tf.random.set_seed(123456)  # Set seed for consistent results


##################################################################################################
#   The most common type of regularization is L2, also called simply “weight decay,”
#   with values often on a logarithmic scale between 0 and 0.1, such as 0.1, 0.001, 0.0001, etc.
#
# Reasonable values of lambda [regularization hyperparameter] range between 0 and 0.1.
#  Decrease LAMBDA  Regularizer  to fight  BIAS
#  Increase LAMBDA  to fight VARIANCE
#
#  regLambda = 10    # 0.01    Lambda for Regularization equation

# Which Layer to user on the Hidden Layers
#   Output Layer must be Softmax with 3 outputs per the project data
#
ACTIVATION_HIDDEN = 'linear'    # Our Hidden Layer Type


model = Sequential(
    [
        ### START STRUCTURE DEFINITION HERE ###

        tf.keras.Input(shape=(6986,)),


        #
        #  relu  == Rectified Linear Unit
        #  linear = Regression
        #  sigmoid == binary
        #
        Dense(512, activation=ACTIVATION_HIDDEN, name='layer1'),
        Dense(256, activation=ACTIVATION_HIDDEN, name='layer2' ),
        Dense(128, activation=ACTIVATION_HIDDEN, name='layer3' ),
        Dense(64, activation=ACTIVATION_HIDDEN, name='layer4' ),
        Dense(32, activation=ACTIVATION_HIDDEN, name='layer5'),
        Dense(16, activation=ACTIVATION_HIDDEN, name='layer6'),
        Dense(3, activation='softmax', name='layer7' )


        # Dense(64, activation=ACTIVATION_HIDDEN, name='layer1'),
        # Dense(32, activation=ACTIVATION_HIDDEN, name='layer2'),
        # Dense(16, activation=ACTIVATION_HIDDEN, name='layer3'),
        # Dense(8, activation=ACTIVATION_HIDDEN, name='layer4'),
        # Dense(3, activation='softmax', name='layer5')


        #    model.add(Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        #  model.add(Conv2D(32, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        #  model.add(LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

        # Dense(64, activation=ACTIVATION_HIDDEN, name='layer1', kernel_regularizer=regularizers.L2(regLambda)),
        # Dense(32, activation=ACTIVATION_HIDDEN, name='layer2', kernel_regularizer=regularizers.L2(regLambda)),
        # Dense(16, activation=ACTIVATION_HIDDEN, name='layer3', kernel_regularizer=regularizers.L2(regLambda)),
        # Dense(8, activation=ACTIVATION_HIDDEN, name='layer4', kernel_regularizer=regularizers.L2(regLambda)),
        # Dense(3, activation='linear', name='layer5', kernel_regularizer=regularizers.L2(regLambda))


        # Dense(2048, activation='relu', name='layer1' ),
        # Dense(1024, activation='relu', name='layer2' ),
        # Dense(64, activation='relu', name='layer3'),
        # Dense(32, activation='relu', name='layer4'),
        # Dense(3, activation='linear', name='layer5' )   # Referred to as 'No Activation'

        ### END DEFINITION HERE ###
    ], name="pennwick_model"
)

model.summary()


#   Add the Loss function  and the Gradient Descent Learning Rate
#
model.compile(

    # loss=tf.keras.losses.BinaryCrossentropy(),    # Used for 'sigmoid'
    #   loss=MeanSquaredError()

    #  SparseCategoricalCrossentropy used for 'Softmax'
    #   from_logits=True  used with 'linear'
    #
    loss=tf.keras.losses.SparseCategoricalCrossentropy( from_logits=False ),

    #  Adam == Adaptive Moment Estimation replaces
    #    Gradient Descent as our optimization method
    #      initial learning/jump rate set to 0.001
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 ),
)

# Now FIT the model to our data
#
history = model.fit(
    X_TRAIN_SplitDF, Y_TRAIN_SplitDF,
    epochs=EPOCH_COUNT
)


# Gather the trained parameters from the output layer
#    l1 = model.get_layer("layer5")
#     W1,b1 = l1.get_weights()

print( "Model is Fit and ready to predict..." )

#
###################################################################################
#  Performance & Evaluation of the Model
#

def AssessErrors(model,X,y):

    # Use X to predict on 'model'

    print("\n##### PREDICT from X on model...")
    predictions = model.predict(X)
    yhat = np.argmax(predictions, axis=1)

    # print("yhat COUNT == ", len( yhat ))
    #  print("y COUNT == ", y.shape )
    # print("y head == ", y.head)

    #  For Each Y ...
    #    Also compute the research statistics - truePositive, etc.
    #
    ctr = 0
    origPositivityRate = predPositivityRate = errors = 0
    truePositive = trueNegative = falsePositive = falseNegative = 0
    df = y.reset_index()  # make sure indexes pair with number of rows

    for index, yp in y.iterrows():

        if ( yp['status_group'] == yhat[ctr] ):
            if yp['status_group'] > 0:
                truePositive = truePositive + 1
            else:
                trueNegative = trueNegative + 1
        else:
            errors = errors + 1   # overall error rate

            if yp['status_group'] > 0:
                falsePositive = falsePositive + 1
            else:
                falseNegative = falseNegative + 1

        if (yp['status_group'] > 0):
            origPositivityRate = origPositivityRate + 1
        if ( yhat[ctr] > 0 ):
            predPositivityRate = predPositivityRate + 1

        ctr = ctr + 1

    # print (  truePositive, falsePositive)
    #  print(  falseNegative, trueNegative)

    precision = truePositive / ( truePositive + falsePositive )
    recall = truePositive / (truePositive + falseNegative )
    print( 'PRECISION == ', precision, '  -  RECALL == ', recall )
    print('ORIG Positivity % == ', (origPositivityRate / ctr) , ' -   PREDICTION Positivity % == ', predPositivityRate / ctr  )

    #
    # doo = yhat != y[:,0]
    #  get only the predictions (yhat) that don't match Groundtruth (y)
    # alt_idxs = np.where(yhat != y[:,0])[0]
    # alt_idxs = np.where(yhat != y)
    # alt_idxs = np.where(yhat == y[:, 0])[0]

    # Return total number of errors
    return(errors)


###################################################################
#  Calculate our ERROR rate on the TRAINING dataset
#
tmp1 = AssessErrors(model,X_TRAIN_SplitDF,Y_TRAIN_SplitDF)
len1 = len(X_TRAIN_SplitDF)
print( f"TRAIN DATASET provides {tmp1} errors out of {len1} predictions  \nfor { round(((len1-tmp1)/len1), 6) } Accuracy and { round((tmp1/len1), 6) } ERROR Rate")


###################################################################
#  Calculate our internal ERROR rate on the VALIDATION dataset
#

tmp1 = AssessErrors(model,X_VALIDATE_SplitDF,Y_VALIDATE_SplitDF)
len1 = len(X_VALIDATE_SplitDF)
print( f"VALIDATION DEV SET provides {tmp1} errors out of {len1} predictions \nfor { round(((len1-tmp1)/len1), 6) } Accuracy and { round((tmp1/len1), 6) } ERROR Rate" )


#########################################################################
#  PREDICT from the TEST set now to prepare the Submission file
#

print("\n##### PREDICT from  TEST   on model for Submission file...")
predictions = model.predict(test_X)  # prediction

predNumpy = pd.DataFrame( predictions )
print("TEST   PREDICTION  Yields  shape ==  ", predNumpy.shape )
# print(f" PREDICTION Yields ==  \n", predNumpy.head() )


#####################################################################
# Now write the submission file in the required format
#
submissionDF = pd.DataFrame(columns =['id', 'status_group'])

# submissionDF['id'] = test_X['id']
submissionDF['id'] = test_X_Saved_IDs
# Determine the highest prediction %
#   on each row  for the final submission prediction
submissionDF['status_group'] = np.argmax(predictions, axis=-1 )

# stats
#
positiveList = np.where(submissionDF['status_group'] > 0)
positiveCount = np.array( positiveList )
print("Prediction POSITIVITY Count == ", positiveCount.size, \
      "  Rate == ", ( positiveCount.size / len(submissionDF) ) )

#    Test random entries for Baseline (Accuracy== .3318  on DD )
#       submissionDF['status_group'] =random.randint(3, size=(len(predictions)) )


# Return the numbers to strings per the submission rules
submissionDF['status_group'].replace([0, 1, 2], ['functional', 'functional needs repair', 'non functional'],
                         inplace=True)

# Write the actual file to disk
submissionDF.to_csv( "pennwick_submission.csv", index=False )
print( "\nWrote Submission file (pennwick_submission.csv) with Shape == ", submissionDF.shape )


# Mission Complete!
##################################################################################
global_later = datetime.now()
print("#####   Tanzania Water Pump Predictions - Total EXECUTION Time =", (global_later - global_now) )
