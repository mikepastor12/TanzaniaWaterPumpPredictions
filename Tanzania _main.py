##########################################################################################
#  Tanzania .py
#
#   Predict the health of water pumps using a Neural Network
#
#   The data for this competition comes from the Taarifa waterpoints dashboard,
#   which aggregates data from the Tanzania Ministry of Water.
#
#   Mike Pastor 10/5/2022
#
# https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/24/

# Local runtime variables
EPOCH_COUNT=12



import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf
from keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid

#  %matplotlib widget
import matplotlib.pyplot as plt
plt.style.use('./pennwick.mplstyle')

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

#############################
# Local runtime and imports
from public_tests import *
from autils import *
np.set_printoptions(precision=2)




#  Track the overall time for training & submission preparation
global_now = datetime.now()
global_current_time = global_now.strftime("%H:%M:%S")
print("##############  Starting up...  - Current Time =", global_current_time)

# load datasets ###########################################################
#
train_X = pd.read_csv('./TrainingSetValues.csv')
train_Y = pd.read_csv('./TrainingSetLABELS.csv')

# Merge the Y onto the X to be sure they are matched
#  This adds 'status_group' LABEL onto each X example row
#
train_X = train_X.merge( train_Y, how='inner', on='id' )
# Used the synchronized LABEL for Y
train_Y = pd.DataFrame( train_X['status_group'], columns=['status_group'] )

################################
#  Load the TEST dataset which is used for the Submission preparation
#
test_X = pd.read_csv('./TestSetValues.csv')
# test_df_sample.info()

print('train_X dataset dimensions are:', train_X.shape)
print('train_Y dataset dimensions are:', train_Y.shape)
print('Test X  dataset dimensions are:', test_X.shape)

#  Take a subset of the columns on both TRAIN and TEST X
#
train_X = train_X[ [ 'id', 'amount_tsh', 'gps_height', 'region_code', 'district_code', 'population', 'construction_year']]
test_X = test_X[ [ 'id', 'amount_tsh', 'gps_height', 'region_code', 'district_code', 'population', 'construction_year' ]]

###########################################
# replacing Y  string values with numerics
#    This is becomes part of our Tanzania Data Dictionary
#
train_Y['status_group'].replace(['functional', 'functional needs repair', 'non functional'],
                        [0, 1, 2], inplace=True)

train_Y = train_Y[ [ 'status_group' ]]


#
# # Distribution of Y
# train_df['label'].value_counts() / train_df.shape[0]
#
#
# ############################################################
# train_df.columns = [i for i in range(0, 785)]
# train = np.array(train_df)
#
# test_df.columns = [i for i in range(0, 784)]
# test = np.array(test_df)
#
# img_rows, img_cols = 28, 28
# num_classes = 10
#
# # num_images = test.shape[0]
# # test_images = test.reshape(num_images, img_rows, img_cols, 1)
# # test_images = test_images / 255
#
# # Setup TRAIN dataset
# #
# Y = train[:, 0]
# X = train[:, 1:]
# num_images = train.shape[0]
# X = X.reshape(num_images, img_rows, img_cols, 1)
# X = X / 255
#
# #  X, Y = prep_Traindata(train)
#
# print('Train  X dataset dimensions are:', X.shape)
# print('Train  Y dataset dimensions are:', Y.shape)

# # Also setup TEST
# # Xtest = test[:, 1:]
# num_images = test.shape[0]
# Xtest = test.reshape(num_images, img_rows, img_cols, 1)
# Xtest = Xtest / 255
#
#
# ########################################################################
# #
# # # Setup Y array
# # y2 = train_df[ 'label' ]
# # y2 = y2.to_numpy()
# # y2 = np.reshape( y2, (len(y2), 1))
# # print('RESHAPED   Shape of y2 dataset is:', y2.shape )
# # # print( '5th element == ',  y2[5,0])
# #
# # # Setup X array
# # X2 = train_df.loc[:, 'pixel0':None]
# # X2 = X2.to_numpy()
# # m2, n2 = X2.shape
# # print('Shape of X2 dataset is:', X2.shape )
# #
# # # Do the same for the TEST X dataset
# # X2test = test_df.loc[:, 'pixel0':None]
# # X2test = X2test.to_numpy()
# # m2test, n2test = X2test.shape
# # print('Shape of TEST  X2 dataset is:', X2test.shape )
#
# # print( X2[0,133])
# # print ('The first column of X2 is: ', X2[:,0] )
# # print ('The first element of y2 is: ', y2[0,0] )
# # print ('The last element of y2 is: ', y2[-1,0] )
# #
# # print ('The shape of X2 is: ' + str(X2.shape))
# # print ('The shape of y2 is: ' + str(y2.shape))
# # print ('m2 == ', m2, '  n2 == ', n2)

#  Now split the TRAINING dataset into a VALIDATION set also
#
####################################################################################
#  TBD   Create a split Training and new Validation from the  merged TRAIN dataset
#
X_TRAIN_SplitDF, X_VALIDATE_SplitDF,  Y_TRAIN_SplitDF, Y_VALIDATE_SplitDF =\
    train_test_split( train_X,  train_Y, test_size=0.3 )

print(f"***********  Train X  set is {len(X_TRAIN_SplitDF)} rows  and Validation set is {len(X_VALIDATE_SplitDF)} rows ")
print(f"***********  Train Y  set is {len(Y_TRAIN_SplitDF)} rows  and Validation set is {len(Y_VALIDATE_SplitDF)} rows ")


############################################################################################
# Setup the Layer structure for our NN with a Sequential model
# Andrew's example
#    32 / 16 / 8 / 4 / 12 / 10
#
# Summary of steps
#     Get more Training examples  -> Fixes High Variance
#       Try smaller set of features  ->  Fixes High Variance
#       Try Additional features ->  Fixes High Bias
#       Try adding polynomial features (x squared, etc) -> Fixes High Bias
#       Decrease Lambda ->  Fixes High Bias
#       Increase Lambda ->   Fixes High Variance

tf.random.set_seed(123456)  # Set seed for consistent results

#  Decrease Regularizer  to fight  BIAS  increase to fight VARIANCE
regLambda = 0.01    # Lambda for Regularization equation
#
# tf.keras.Input(shape=(784,)),
# Dense(32, activation='relu', name='layer1', kernel_regularizer=regularizers.L2(regLambda)),
# Dense(16, activation='relu', name='layer2', kernel_regularizer=regularizers.L2(regLambda)),
# Dense(8, activation='relu', name='layer3', kernel_regularizer=regularizers.L2(regLambda)),
# Dense(4, activation='relu', name='layer4', kernel_regularizer=regularizers.L2(regLambda)),
# Dense(12, activation='relu', name='layer5', kernel_regularizer=regularizers.L2(regLambda)),
# Dense(10, activation='linear', name='layer6', kernel_regularizer=regularizers.L2(regLambda))

model = Sequential(
    [
        ### START STRUCTURE DEFINITION HERE ###

        tf.keras.Input(shape=(7,)),
        # Dense(256, activation='sigmoid', name='layer1' ),

        Dense(64, activation='relu', name='layer1' ),
        Dense(32, activation='relu', name='layer2' ),
        Dense(3, activation='linear', name='layer3' )

        # tf.keras.Input(shape=(784,)),
        # Dense(32, activation='relu', name='layer1' ),
        # Dense(16, activation='relu', name='layer2' ),
        # Dense(8, activation='relu', name='layer3' ),
        # Dense(4, activation='relu', name='layer4' ),
        # Dense(12, activation='relu', name='layer5'),
        # Dense(10, activation='linear', name='layer6' )

        ###   TBD  tf.keras.Input(shape=(3,)),

        #
        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2, 2),
        #
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dense(10, activation='softmax')

        ### END DEFINITION HERE ###
    ], name="pennwick_model"
)

model.summary()

#   Add the Loss function  and the Gradient Descent Learning Rate
#
model.compile(

    #  loss=BinaryCrossentrophy()
    loss=tf.keras.losses.SparseCategoricalCrossentropy( from_logits=True ),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

# 100 Epochs as default
# history = model.fit(
#     X_TRAIN_SplitDF, Y_TRAIN_SplitDF,
#     epochs=100
# )

history = model.fit(
    X_TRAIN_SplitDF, Y_TRAIN_SplitDF,
    epochs=EPOCH_COUNT
)


print( "Model is Fit and ready to predict...")

predictions = model.predict(test_X)  # prediction

predNumpy = pd.DataFrame( predictions )
print(f" PREDICTION shape ==  ", predNumpy.shape )
print(f" PREDICTION Yields ==  ", predNumpy.head() )

print(f" PREDICTION index VALUE == {np.argmax(predictions[0])}")


#####################################################################
# Now write the submission file in the required format
#
submissionDF = pd.DataFrame(columns =['id', 'status_group'])

submissionDF['id'] = test_X['id']
submissionDF['status_group'] = np.argmax(predictions, axis=-1 )

# Return the numbers to strings per the submission rules
submissionDF['status_group'].replace([0, 1, 2], ['functional', 'functional needs repair', 'non functional'],
                         inplace=True)

submissionDF.to_csv( "pennwick_submission.csv", index=False )

#
# ####################################################################
# print( "Predict an Number ...")
# drawingNumber = 23
# image_of_two2 = X_TRAIN_SplitDF[drawingNumber]
# print( "Image Y LABEL == ", Y_TRAIN_SplitDF[drawingNumber])
# # y2[5,0]
# print( "call  model.predict   ...")
#
# #  num_images = test.shape[0]
# # test_images = test.reshape(num_images, img_rows, img_cols, 1)
# # prediction = model.predict(image_of_two2.reshape(1,784))  # prediction
# prediction = model.predict(image_of_two2.reshape(1, img_rows, img_cols, 1))  # prediction
#
# # print(f" PREDICTION Yields ==  \n{prediction}")
# print(f" PREDICTION index VALUE == {np.argmax(prediction)}")
#
# prediction_p = tf.nn.softmax(prediction)
# # print(f" Try  SOFTMAX Probability vector: \n{prediction_p}")
# # print(f"Total of SOFTMAX predictions: {np.sum(prediction_p):0.3f}")
#
# yhat = np.argmax(prediction_p)
# print(f"SOFTMAX PREDICTION == np.argmax(prediction_p): {yhat}")

######################################################################
def display_errors2(model,X,y):

    # Use X to predict on 'model'
    print("\n##### PREDICT from X on model...")
    predictions = model.predict(X)
    yhat = np.argmax(predictions, axis=1)

    print("yhat COUNT == ", len( yhat ))
    print("y COUNT == ", y.shape )
    # print("y head == ", y.head)

    #  For Each Y ...
    ctr = 0
    errors = 0
    df = y.reset_index()  # make sure indexes pair with number of rows
    for index, yp in y.iterrows():

        # print(yp['status_group'])
        # print(yhat[ctr])
        if ( yp['status_group'] != yhat[ctr] ):
            # print('NO MATCH!!!!!!')
            errors = errors + 1

        ctr = ctr + 1

    #
    # doo = yhat != y[:,0]
    #  get only the predictions (yhat) that don't match Groundtruth (y)
    #
    # alt_idxs = np.where(yhat != y[:,0])[0]
    # alt_idxs = np.where(yhat != y)
    # alt_idxs = np.where(yhat == y[:, 0])[0]

    if errors == 0:
        print("no errors found")
    else:
        print( f" {errors} Errors Found ")


    # Return number of errors
    return(errors)


###################################################################
#  Calculate our ERROR rate on the TRAINING dataset
#
tmp1 = display_errors2(model,X_TRAIN_SplitDF,Y_TRAIN_SplitDF)
len1 = len(X_TRAIN_SplitDF)
print( f"TRAIN DATASET provides {tmp1} errors out of {len1} images  \nfor { round(((len1-tmp1)/len1), 6) } Accuracy and { round((tmp1/len1), 6) } ERROR Rate")

###################################################################
#  Calculate our internal ERROR rate on the VALIDATION dataset
#
tmp1 = display_errors2(model,X_VALIDATE_SplitDF,Y_VALIDATE_SplitDF)
len1 = len(X_VALIDATE_SplitDF)
print( f"VALIDATION DEV SET provides {tmp1} errors out of {len1} images \nfor { round(((len1-tmp1)/len1), 6) } Accuracy and { round((tmp1/len1), 6) } ERROR Rate" )

exit(-9)




##########################################################################################################
# Predict using the test dataset
# prediction = model.predict(Xtest.reshape(len(Xtest), 784))
prediction = model.predict(Xtest)
prediction_p = tf.nn.softmax(prediction)
yhat = np.argmax(prediction_p)

print( "predictions on TEST == ", len(prediction_p))

####################################################
# Write the Kaggle submission file
#
FILENAME = "submission.csv"
imageID = np.zeros( len(prediction_p) )
imageID = imageID.reshape( len(prediction_p) , 1)
myPrediction = np.zeros( len(prediction_p)  )
myPrediction = myPrediction.reshape( len(prediction_p) , 1)

print( "About to write:  ", len(prediction_p), "  Predictions to ", FILENAME )
ctr = 0
while ctr < len(prediction_p):

    imageID[ctr] = int( ctr+1 )
    tmpPred = np.argmax(prediction_p[ctr])
    myPrediction[ctr] = tmpPred
    # print("PREDICT: ", tmpPred)
    ctr = ctr + 1

imageID = imageID.astype(int)
myPrediction = myPrediction.astype(int)
all_data = pd.DataFrame(np.hstack((imageID, myPrediction)), columns =['ImageId', 'Label'])
# print( "About to  write:  ", all_data.shape, all_data[:12] )

submissionDF = all_data
print( "About to really write:  ", submissionDF.shape, submissionDF[:6] )

submissionDF.to_csv( FILENAME, index=False )
print( "Successfully Wrote ", FILENAME )

# Mission Complete!

##################################################################################
global_later = datetime.now()
global_later_time = global_later.strftime("%H:%M:%S")
# print("DONE - train_data.csv  - Current Time =", global_later_time)
print("MNIST Digit Recognition - Total EXECUTION Time =", (global_later - global_now) )

print("Done...")