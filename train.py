import pandas as pd
import numpy as np
import tensorflow as tf
import os 
import time
import progressbar
import argparse
import json
import sys
import shutil

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("nEmbeds", help = "Number of latent factors", type = int)

    # Optional arguments
    parser.add_argument("-f", "--file", help = "File", type = str, default = '100000.csv')
    parser.add_argument("-o", "--outputPath", help = "Output path", type = str, default = 'wals_trained')
    parser.add_argument("-i", "--inputPath", help = "Input path", type = str, default = 'data')

    # Parse arguments
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = parseArguments()
    #shutil.rmtree(args.outputPath, ignore_errors = True)
    INPUT_PATH = args.inputPath + '/'
    
    print(str(args.nEmbeds) + " latent factors are going to be estimated.") 

    print("Loading all user/item interactions from " + INPUT_PATH + args.file)
    # possible improvement: use of a BigTable
    df = pd.read_csv(INPUT_PATH + args.file,
                     header = None,
                     names = ['userId', 'choice', 'itemId'],
                     dtype = {'userId': str, 'itemId': str, 'choice': str})

    res = []

    def job():
        bar = progressbar.ProgressBar()
        print("Assigning ratings to interactions")
        for row in bar(df.to_records()):
            if (row[2] == 'croix'):
                res.append({'userId': row[1], 'rating': -1.0, 'itemId': row[3] + '_image_0'})
                res.append({'userId': row[1], 'rating': -1.0, 'itemId': row[3] + '_image_1'})
                res.append({'userId': row[1], 'rating': -1.0, 'itemId': row[3] + '_image_2'})
                res.append({'userId': row[1], 'rating': -1.0, 'itemId': row[3] + '_image_3'})
            else:
                rating0 = 3.0 if row[2] == 'image_0' else 0.0
                rating1 = 3.0 if row[2] == 'image_1' else 0.0
                rating2 = 3.0 if row[2] == 'image_2' else 0.0
                rating3 = 3.0 if row[2] == 'image_3' else 0.0
                res.append({'userId': row[1], 'rating': rating0, 'itemId': row[3] + '_image_0'})
                res.append({'userId': row[1], 'rating': rating1, 'itemId': row[3] + '_image_1'})
                res.append({'userId': row[1], 'rating': rating2, 'itemId': row[3] + '_image_2'})
                res.append({'userId': row[1], 'rating': rating3, 'itemId': row[3] + '_image_3'})  
            
    job()
        
    df = pd.DataFrame(res)
    
    # possible improvement: use of a SQL database instead of these
    user_mapping = create_mapping(df['userId'], INPUT_PATH + 'users.csv')
    item_mapping = create_mapping(df['itemId'], INPUT_PATH + 'items.csv')

    # map userId and itemId to integers
    df['userId'] = df['userId'].map(user_mapping.get)
    df['itemId'] = df['itemId'].map(item_mapping.get)

    # re-order the columns
    mapped_df = df[['userId', 'itemId', 'rating']]

    NITEMS = np.max(mapped_df['itemId']) + 1
    NUSERS = np.max(mapped_df['userId']) + 1
    mapped_df['rating'] = np.round(mapped_df['rating'].values, 2)
    print('{} items, {} users, {} interactions'.format(NITEMS, NUSERS, len(mapped_df)))

    grouped_by_items = mapped_df.groupby('itemId')
    with tf.python_io.TFRecordWriter(INPUT_PATH + 'users_for_item') as ofp:
        for item, grouped in grouped_by_items:
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'key': tf.train.Feature(int64_list = tf.train.Int64List(value = [item])),
                        'indices': tf.train.Feature(int64_list = tf.train.Int64List(value = grouped['userId'].values)),
                        'values': tf.train.Feature(float_list = tf.train.FloatList(value = grouped['rating'].values))
                    }
                )
            )
            ofp.write(example.SerializeToString())
            
    grouped_by_users = mapped_df.groupby('userId')
    with tf.python_io.TFRecordWriter(INPUT_PATH + 'items_for_user') as ofp:
        for user, grouped in grouped_by_users:
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'key': tf.train.Feature(int64_list = tf.train.Int64List(value = [user])),
                        'indices': tf.train.Feature(int64_list = tf.train.Int64List(value = grouped['itemId'].values)),
                        'values': tf.train.Feature(float_list = tf.train.FloatList(value = grouped['rating'].values))
                    }
                )
            )
            ofp.write(example.SerializeToString())
    
    train_and_evaluate({
        'output_dir': args.outputPath + "/",
        'input_path': INPUT_PATH,
        'num_epochs': 0.001,
        'nitems': NITEMS,
        'nusers': NUSERS,
        'batch_size': 512,
        'n_embeds': int(sys.argv[1])
    })
    d = dict()
    d['n_embeds'] = args.nEmbeds
    d['output_dir'] = args.outputPath
    d['nitems'] = NITEMS
    d['nusers'] = NUSERS
    with open(args.outputPath + '/wals_parameters.txt', 'w') as outfile:
        json.dump(d, outfile)
    print("Finished training. Model has been saved to " + args.outputPath + " and parameters have been saved to " + INPUT_PATH + "wals_parameters.txt")