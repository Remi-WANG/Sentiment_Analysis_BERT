# importing the required packages

import numpy as np
import pandas as pd
from sklearn.svm import SVC
import torch
import transformers as ppb
from sklearn.preprocessing import LabelEncoder
import warnings
from transformers import BertModel, BertTokenizer
warnings.filterwarnings('ignore')

import os


# datadir2 = '/home/jupyter/NLP/resource/'
# vocab =  datadir2 + "vocab.txt"
# model_file =  datadir2 + "config.json"

# A pretrained BERT model 

# old code
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-large-uncased')
# tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
# #model = model_class.from_pretrained(pretrained_weights)


#################################CLOUD REFERRAL CODE #################


# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)


# cwd = os.getcwd()

# backOne = os.path.dirname(cwd)

# model_path = backOne + "/resource"

# print(model_path)

#################################CLOUD REFERRAL CODE #################

BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )

model_path = BASE_DIR + "/resources"

txt_path = BASE_DIR + "/resources/vocab.txt"

#new code
model = BertModel.from_pretrained(model_path)
tokenizer=BertTokenizer.from_pretrained(txt_path)


# defining the label encoder

label_encoder = LabelEncoder()

class Classifier:
    """The Classifier"""
    ##########################################
    def init(self):
        pass

    # Get embeddings from BERT
    def embeddings(self, dataset):
        
        # Tokenize text - break the review into word and subwords in the format that suits BERT
        tokenized = dataset['review'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


        # Since BERT will process our examples all at once
        # Pad text to the maximum length in the data set
        # so we can represent the input as one 2-d array.
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

        # Create mask
        # A variable to tell BERT to ignore (mask) the padding we've added
        attention_mask = np.where(padded != 0, 1, 0)

        # Convert to a tensor
        input_ids = torch.tensor(padded)  
        attention_mask = torch.tensor(attention_mask)

        # Generate embeddings
        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        # Take only the sentence embedding
        features = last_hidden_states[0][:,0,:].numpy()

        return features


    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""

        # Get embeddings from BERT
        train = pd.read_csv(trainfile, sep='\t', names=['sentiment','aspect_category','word','position','review'])
        train_features = self.embeddings(train)


        # label encoding for the sentiment
        train['integer_sentiment'] = label_encoder.fit_transform(train.sentiment)

		# Also taking the labels as interger setiments
        labels = train['integer_sentiment']
		

		# Model training
        clf = SVC()
        clf.fit(train_features, labels)
        self.clf = clf

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """

        # Get embeddings from BERT
        test = pd.read_csv(datafile, sep='\t', names=['sentiment','aspect_category','word','position','review'])
        test_features = self.embeddings(test)


		# Model prediction
        pred = self.clf.predict(test_features)

		# converting back to sentiment
        sentiment = label_encoder.inverse_transform(pred)

        return sentiment
