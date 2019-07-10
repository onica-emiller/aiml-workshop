import sys
import os
import json
import argparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib

#-----------------------------------------------------------------------
#  Preprocessing functions.  These will isolate the data between the
#  leading and trailing edges and interpolate to a fixed length signal
#  in preparation for passing to the model for inference.
#-----------------------------------------------------------------------

def prepare_data_touch(x,y):
        
    #-- define a threshold to determine leading/trailing exceedance edge
    my = np.median(y)
    thresh = 0.75 * my
    
    #-- find leading/trailing edges
    x0 = np.min(x[y<thresh]) if np.any(y<thresh) else np.min(x)
    xN = np.max(x[y<thresh]) if np.any(y<thresh) else np.max(x)
    
    #-- start and end a bit before/after edges
    x0 -= 250
    xN += 250
    
    xn = (x-x0) / float(xN-x0)
    yn = y[(xn>=0) & (xn<=1)]
    xn = xn[(xn>=0) & (xn<=1)]
    
    newx = np.arange(500)*0.002
    newy = np.interp(newx,xn,yn)
    return newy

def prepare_data_button(x,y):

    #-- define a threshold to determine leading/trailing exceedance edge
    thresh = 0.5
    
    #-- find leading/trailing edges
    x0 = np.min(x[y>thresh]) if np.any(y>thresh) else np.min(x)
    xN = np.max(x[y>thresh]) if np.any(y>thresh) else np.max(x)
    
    #-- start and end a bit before/after edges
    x0 -= 250
    xN += 250
    
    xn = (x-x0) / float(xN-x0)
    yn = y[(xn>=0) & (xn<=1)]
    xn = xn[(xn>=0) & (xn<=1)]
    
    newx = np.arange(500)*0.002
    newy = np.interp(newx,xn,yn)
    return newx,newy

#-----------------------------------------------------------------------
#  Use the appropriate preparation function for the primary sensor used
#-----------------------------------------------------------------------

def prepare_data(x,y):
    return prepare_data_button(x,y)

#-----------------------------------------------------------------------
#  Perform all preprocessing of data (takes place of read_data in
#  notebooks, and only returns y values.
#-----------------------------------------------------------------------

def preprocess(rawdata):
    
    #-------------------------------------------------------------------
    # Raw data should come in as a list of x,y pairs.
    # Convert to two arrays for x and y.
    #-------------------------------------------------------------------
    rawx,rawy = zip(*rawdata)

    #-- subtract off initial timestamp so x starts at zero
    x = np.array(rawx)-rawx[0]
    y = np.array(rawy)

    xp, yp = prepare_data(x,y)

    return yp

#-----------------------------------------------------------------------
#  Implemented to override default function and apply pre-processing.
#-----------------------------------------------------------------------

def input_fn(input_data, content_type):

    if(content_type == 'application/json'):
        if(type(input_data) is str):
            raw = json.loads(input_data)
        elif(type(input_data) is bytes):
            raw = json.loads(input_data.decode('utf-8'))
        else:
            raise ValueError("Unsupported serialization: {}".format(type(input_data)))
    elif(content_type == 'application/octet-stream'):
        try:
            raw = json.loads(input_data.decode('utf-8'))
        except ValueError:
            raise ValueError("Could not decode byte stream to JSON")
    else:
        raise ValueError("{} not supported by script!".format(content_type))

    rawdata = raw['records']
    
    #-- wrap numpy result in a list for compatibility with predict_fn
    data = [preprocess(rawdata)]

    return data
        
#-----------------------------------------------------------------------
#  
#-----------------------------------------------------------------------

def output_fn(prediction, accept):
    result = {'prediction': prediction.tolist()}
    return json.dumps(result).encode('utf-8')
    
#-----------------------------------------------------------------------
#  
#-----------------------------------------------------------------------

def model_fn(model_dir):
    km = joblib.load(os.path.join(model_dir, "model.joblib"))
    return km

#-----------------------------------------------------------------------
#  The main guard code is invoked when SageMaker calls .fit on an
#  SKLearn model where this file is the designated entry_point.
#-----------------------------------------------------------------------

if __name__=='__main__':

    #-- pull environment variables if running in SageMaker, empty otherwise
    defaults = {}
    for key in ['SM_OUTPUT_DATA_DIR', 'SM_MODEL_DIR', 'SM_CHANNEL_TRAIN']:
        defaults[key] = os.environ[key] if key in os.environ else ''
    
    parser = argparse.ArgumentParser()

    # Only one hyperparameter for KMeans
    parser.add_argument('--n_clusters', type=int, default=2)

    # Options to override SageMaker environment variables.
    parser.add_argument('--output-data-dir', type=str, default=defaults['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=defaults['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=defaults['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    data = []
    for basename in os.listdir(args.train):
        filename = os.path.join(args.train, basename)
        with open(filename, 'rb') as fh:
            d = input_fn(fh.read(), 'application/json')
        data.append(d[0])

    km = KMeans(n_clusters=args.n_clusters)
    km.fit(data)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(km, os.path.join(args.model_dir, "model.joblib"))

    # for d in data:
    #     out = output_fn(km.predict([d]), 'application/json')
    #     print(out)
