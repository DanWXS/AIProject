#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of the required functions, remove non-required functions, and add your own functions.

################################################################################
#
# Imported functions and variables
#
################################################################################

# Import functions. These functions are not required. You can change or remove them.
from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)

################################################################################
#
# Training model function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    scored = ['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002', '39732003',
              '164909002',
              '251146004', '698252002', '10370003', '284470004', '427172004', '164947007', '111975006', '164917005',
              '47665007', '59118001',
              '427393009', '426177001', '426783006', '427084000', '63593006', '164934002', '59931005', '17338001']

    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract the classes from the dataset.
    print('Extracting classes...')

    classes = set()
    # for header_file in header_files:
    #     header = load_header(header_file)
    #     classes |= set(get_labels(header))
    # if all(is_integer(x) for x in classes):
    #     classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    # else:
    #     classes = sorted(classes) # Sort classes alphanumerically if not numbers.
    # num_classes = len(classes)

    classes = sorted(scored, key=lambda x: int(x))# Sort classes numerically if numbers.
    num_classes = len(classes)
    # Extract the features and labels from the dataset.
    print('Extracting features and labels...')

    data = np.zeros((num_recordings, 14), dtype=np.float32) # 14 features: one feature for each lead, one feature for age, and one feature for sex
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool_) # One-hot encoding of classes

    for i in range(num_recordings):
        # print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])

        current_labels = get_labels(header)
        for label in current_labels:
            if label in classes:  # np.where(scored == label):
                j = classes.index(label)
                labels[i, j] = 1
                # Get age, sex and root mean square of the leads.
                age, sex, rms = get_features(header, recording, twelve_leads)
                data[i, 0:12] = rms
                data[i, 12] = age/100
                data[i, 13] = sex

    data = data[~np.all(data == 0, axis=1)]
    labels = labels[~np.all(labels == 0, axis=1)]

    s0 = sum(np.isnan(data[:, 12]))
    indecies = np.argwhere(np.isnan(data[:, 12]))
    data = np.delete(data, indecies, 0)
    labels = np.delete(labels, indecies, 0)
    #
    # np.save('Train_data_28_2.npy', data)
    # np.save('Train_labels_28_2.npy', labels)
    # np.save('Train_classes_28_2.npy', classes)

    # data = np.load('Train_data_28_2.npy')
    # labels = np.load('Train_labels_28_2.npy')
    # classes = np.load('Train_classes_28_2.npy')

    # Train a model for each lead set.
    for leads in lead_sets:
        print('Training model for {}-lead set: {}...'.format(len(leads), ', '.join(leads)))

        # Extract the features for the model.
        feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
        features = data[:, feature_indices]

        # Train the model.
        imputer = SimpleImputer().fit(features)
        features = imputer.transform(features)
        # features[:, 12] = features[:, 12] / 104

        # lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)
        # if leads == twelve_leads:
        #     # Define parameters for random forest classifier.
        #     n_estimators = 2  # Number of trees in the forest.
        #     max_leaf_nodes = 3  # Maximum number of leaf nodes in each tree.
        #     random_state = 0  # Random state; set for reproducibility.
        # else:
        #     # Define parameters for random forest classifier.
        #     n_estimators = 1  # Number of trees in the forest.
        #     max_leaf_nodes = 3  # Maximum number of leaf nodes in each tree.
        #     random_state = 0  # Random state; set for reproducibility.

        # Define parameters for random forest classifier.
        n_estimators = 1  # Number of trees in the forest.
        max_leaf_nodes = 2  # Maximum number of leaf nodes in each tree.
        random_state = 0  # Random state; set for reproducibility.

        # classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)
        classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state).fit(features, labels)

        # Save the model.
        save_model(model_directory, leads, classes, imputer, classifier)

################################################################################
#
# Running trained model function
#
################################################################################

# Run your trained model. This function is *required*. Do *not* change the arguments of this function.
def run_model(model, header, recording):
    classes = model['classes']
    leads = model['leads']
    imputer = model['imputer']
    classifier = model['classifier']

    # Load features.
    num_leads = len(leads)
    data = np.zeros(num_leads+2)
    age, sex, rms = get_features(header, recording, leads)
    data[0:num_leads] = rms
    data[num_leads] = age/100
    data[num_leads+1] = sex

    # Impute missing data.
    features = data.reshape(1, -1)
    features = imputer.transform(features)

    # Predict labels and probabilities.
    labels = classifier.predict(features)
    labels = np.asarray(labels, dtype=np.int_)[0]

    probabilities = classifier.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    return classes, labels, probabilities

################################################################################
#
# File I/O functions
#
################################################################################

# Save a trained model. This function is not required. You can change or remove it.
def save_model(model_directory, leads, classes, imputer, classifier):
    d = {'leads': leads, 'classes': classes, 'imputer': imputer, 'classifier': classifier}
    filename = os.path.join(model_directory, get_model_filename(leads))
    joblib.dump(d, filename, protocol=0)

# Load a trained model. This function is *required*. Do *not* change the arguments of this function.
def load_model(model_directory, leads):
    filename = os.path.join(model_directory, get_model_filename(leads))
    return joblib.load(filename)

# Define the filename(s) for the trained models. This function is not required. You can change or remove it.
def get_model_filename(leads):
    return 'model_' + '-'.join(sort_leads(leads)) + '.sav'

################################################################################
#
# Feature extraction function
#
################################################################################

# Extract features from the header and recording. This function is not required. You can change or remove it.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    recording = choose_leads(recording, header, leads)

    # Pre-process recordings.
    adc_gains = get_adc_gains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

    return age, sex, rms



# data_directory = 'D:/Challenge2021/Data/Train_data'
# model_directory = 'D:/Challenge2021/Model'
# training_code(data_directory, model_directory)