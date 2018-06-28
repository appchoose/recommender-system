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
from recsys.blackbox import *
from recsys.tfutils import *
from recsys.sqlite import *

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("nEmbeds", help = "Number of latent factors", type = int)

    # Optional arguments
    parser.add_argument("-o", "--outputPath", help = "Output path", type = str, default = 'wals_trained')
    parser.add_argument("-bs", "--batchSize", help = "Batch size", type = int, default = 512)
    parser.add_argument("-e", "--epochs", help = "Number of epochs", type = int, default = 1)
    parser.add_argument("-i", "--nInteractions", help = "Number of interactions", type = int)

    # Parse arguments
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = parseArguments()
    
    if os.path.exists(args.outputPath):
        print("Deleting {}...".format(args.outputPath))
        shutil.rmtree(args.outputPath, ignore_errors = True)
    
    print("{} latent factors are going to be estimated.".format(args.nEmbeds)) 
    
    if args.nInteractions is None:
        sql_q = "SELECT * FROM rating"
    else:
        sql_q = "SELECT * FROM rating LIMIT " + str(args.nInteractions)
    
    # possible improvement: use of a BigTable
    conn = sqlite3.connect("db")
    df = pd.read_sql_query(con = conn, sql = sql_q)
    user_mapping = pd.read_sql_query(con = conn, sql = "SELECT * FROM user")
    item_mapping = pd.read_sql_query(con = conn, sql = "SELECT * FROM item")
    conn.close()
    
    # map userId and itemId to integers
    df['itemId'] = df['itemId'].map(item_mapping.set_index('item_str')['item_int'].to_dict().get)
    df['userId'] = df['userId'].map(user_mapping.set_index('user_str')['user_int'].to_dict().get)

    NITEMS = np.max(df['itemId']) + 1
    NUSERS = np.max(df['userId']) + 1
    df['rating'] = np.round(df['rating'].values, 2)
    print('Total: {} items, {} users, {} interactions'.format(NITEMS, NUSERS, len(df)))

    write_tf_records(df, by = "itemId", output = "items_for_user")
    write_tf_records(df, by = "userId", output = "users_for_item")
    
    train_and_evaluate({
        'output_dir': args.outputPath + "/",
        'num_epochs': args.epochs,
        'nitems': NITEMS,
        'nusers': NUSERS,
        'batch_size': args.batchSize,
        'n_embeds': args.nEmbeds
    })

    d = dict()
    d['n_embeds'] = args.nEmbeds
    d['output_dir'] = args.outputPath
    d['nitems'] = NITEMS
    d['nusers'] = NUSERS

    with open(args.outputPath + '/wals_parameters.txt', 'w') as outfile:
        json.dump(d, outfile)
        
    print("Finished training. Model has been saved to {} and parameters have been saved to wals_parameters.txt".format(args.outputPath))