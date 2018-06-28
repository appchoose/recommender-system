import pandas as pd
import numpy as np
import tensorflow as tf
import os 
from tensorflow.python.lib.io import file_io
from tensorflow.contrib.factorization import WALSMatrixFactorization
from tensorflow.contrib.learn.python.learn import learn_runner

def read_dataset(mode, args):
    
    def decode_example(protos, vocab_size):
        features = {'key': tf.FixedLenFeature([1], tf.int64),
                    'indices': tf.VarLenFeature(dtype = tf.int64),
                    'values': tf.VarLenFeature(dtype = tf.float32)}
        parsed_features = tf.parse_single_example(protos, features)
        values = tf.sparse_merge(parsed_features['indices'], 
                                 parsed_features['values'], 
                                 vocab_size = vocab_size)
        # Save key to remap after batching
        key = parsed_features['key']
        decoded_sparse_tensor = tf.SparseTensor(indices = tf.concat([values.indices, [key]], axis = 0), 
                                                values = tf.concat([values.values, [0.0]], axis = 0), 
                                                dense_shape = values.dense_shape)
        return decoded_sparse_tensor
  
    def remap_keys(sparse_tensor):
        # Current indices of our SparseTensor that we need to fix
        bad_indices = sparse_tensor.indices
        # Current values of our SparseTensor that we need to fix
        bad_values = sparse_tensor.values 
  
        # Group by the batch_indices and get the count for each  
        size = tf.segment_sum(data = tf.ones_like(bad_indices[:,0], dtype = tf.int64), 
                              segment_ids = bad_indices[:,0]) - 1
        # The number of batch_indices (this should be batch_size unless it is a partially full batch)
        length = tf.shape(size, out_type = tf.int64)[0]
        # Finds the cumulative sum which we can use for indexing later
        cum = tf.cumsum(size)
        # The offsets between each example in the batch due to our concatenation of the keys in the decode_example method
        length_range = tf.range(start = 0, limit = length, delta = 1, dtype = tf.int64)
        # Indices of the SparseTensor's indices member of the rows we added by the concatentation of our keys in the decode_example method
        cum_range = cum + length_range

        # The keys that we have extracted back out of our concatentated SparseTensor
        gathered_indices = tf.squeeze(tf.gather(bad_indices, cum_range)[:,1])

        # The enumerated row indices of the SparseTensor's indices member
        sparse_indices_range = tf.range(tf.shape(bad_indices, out_type = tf.int64)[0], dtype = tf.int64)

        # We want to find here the row indices of the SparseTensor's indices member that are of our actual data and not the concatenated
        # rows
        # So we want to find the intersection of the two sets and then take the opposite of that
        x = sparse_indices_range
        s = cum_range

        # Number of multiples we are going to tile x, which is our sparse_indices_range
        tile_multiples = tf.concat([tf.ones(tf.shape(tf.shape(x)), dtype = tf.int64), 
                                    tf.shape(s, out_type = tf.int64)], axis = 0)
        # Expands x, our sparse_indices_range, into a rank 2 tensor and then multiplies the rows by 1 (no copying) and the columns by the
        # number of examples in the batch
        x_tile = tf.tile(tf.expand_dims(x, -1), tile_multiples)
        # Essentially a vectorized logical or, that we then negate
        x_not_in_s = ~tf.reduce_any(tf.equal(x_tile, s), -1)

        # The SparseTensor's indices that are our actual data by using the boolean_mask we just made above applied to the entire indices
        # member of our SparseTensor
        selected_indices = tf.boolean_mask(tensor = bad_indices, mask = x_not_in_s, axis = 0)
        # Apply the same boolean_mask to the entire values member of our SparseTensor to get the actual values data
        selected_values = tf.boolean_mask(tensor = bad_values, mask = x_not_in_s, axis = 0)

        # Need to replace the first column of our selected_indices with keys, so we first need to tile our gathered_indices
        tiling = tf.tile(input = tf.expand_dims(gathered_indices[0], -1), multiples = tf.expand_dims(size[0] , -1))
    
        # We have to repeatedly apply the tiling to each example in the batch
        # Since it is jagged we cannot use tf.map_fn due to the stacking of the TensorArray, so we have to create our own custom version
        def loop_body(i, tensor_grow):
            return i + 1, tf.concat(values = [tensor_grow, tf.tile(input = tf.expand_dims(gathered_indices[i], -1), 
                                                                   multiples = tf.expand_dims(size[i], -1))], axis = 0)

        _, result = tf.while_loop(lambda i, tensor_grow: i < length, loop_body, [tf.constant(1, dtype = tf.int64), tiling])
    
        # Concatenate tiled keys with the 2nd column of selected_indices
        selected_indices_fixed = tf.concat([tf.expand_dims(result, -1), tf.expand_dims(selected_indices[:, 1], -1)], axis = 1)
    
        # Combine everything together back into a SparseTensor
        remapped_sparse_tensor = tf.SparseTensor(indices = selected_indices_fixed, 
                                                 values = selected_values, 
                                                 dense_shape = sparse_tensor.dense_shape)
        return remapped_sparse_tensor

    def parse_tfrecords(filename, vocab_size):
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
        else:
            num_epochs = 1 # end-of-input after this
    
        files = tf.gfile.Glob(filename)
    
        # Create dataset from file list
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(lambda x: decode_example(x, vocab_size))
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(args['batch_size'])
        dataset = dataset.map(lambda x: remap_keys(x))
        return dataset.make_one_shot_iterator().get_next()
  
    def _input_fn():
        features = {
            WALSMatrixFactorization.INPUT_ROWS: parse_tfrecords('items_for_user', args['nitems']),
            WALSMatrixFactorization.INPUT_COLS: parse_tfrecords('users_for_item', args['nusers']),
            WALSMatrixFactorization.PROJECT_ROW: tf.constant(True)
        }
        return features, None
  
    def input_cols():
        return parse_tfrecords('users_for_item', args['nusers'])
    
    return _input_fn

# online prediction returns row and column factors as needed
def create_serving_input_fn(args):
    
    def for_user_embeddings(userId):
        # all items for this user (for user_embeddings)
        items = tf.range(args['nitems'], dtype = tf.int64)
        users = userId * tf.ones([args['nitems']], dtype = tf.int64)
        ratings = 0.1 * tf.ones_like(users, dtype = tf.float32)
        return items, users, ratings, tf.constant(True)
    
    def for_item_embeddings(itemId):
        # all users for this item (for item_embeddings)
        users = tf.range(args['nusers'], dtype = tf.int64)
        items = itemId * tf.ones([args['nusers']], dtype = tf.int64)
        ratings = 0.1 * tf.ones_like(users, dtype = tf.float32)
        return items, users, ratings, tf.constant(False)
    
    def serving_input_fn():
        feature_ph = {
            'userId': tf.placeholder(tf.int64, 1),
            'itemId': tf.placeholder(tf.int64, 1)
        }
            
        (items, users, ratings, project_row) = tf.cond(
            feature_ph['userId'][0] < tf.constant(0, dtype = tf.int64),
            lambda: for_item_embeddings(feature_ph['itemId']),
            lambda: for_user_embeddings(feature_ph['userId'])
        )
        rows = tf.stack( [users, items], axis = 1 )
        cols = tf.stack( [items, users], axis = 1 )
        input_rows = tf.SparseTensor(rows, ratings, (args['nusers'], args['nitems']))
        input_cols = tf.SparseTensor(cols, ratings, (args['nusers'], args['nitems']))
    
        features = {
            WALSMatrixFactorization.INPUT_ROWS: input_rows,
            WALSMatrixFactorization.INPUT_COLS: input_cols,
            WALSMatrixFactorization.PROJECT_ROW: project_row
        }
        return tf.contrib.learn.InputFnOps(features, None, feature_ph)
    
    return serving_input_fn

def train_and_evaluate(args):
    train_steps = int(0.1 + (1.0 * args['num_epochs'] * args['nusers']) / args['batch_size'])
    steps_in_epoch = int(0.1 + args['nusers'] / args['batch_size'])
    print('Will train for {} steps, evaluating once every {} steps'.format(train_steps, steps_in_epoch))
    
    def experiment_fn(output_dir):
        return tf.contrib.learn.Experiment(
            tf.contrib.factorization.WALSMatrixFactorization(
                num_rows = args['nusers'], 
                num_cols = args['nitems'],
                embedding_dimension = args['n_embeds'],
                model_dir = args['output_dir']),
            train_input_fn = read_dataset(tf.estimator.ModeKeys.TRAIN, args),
            eval_input_fn = read_dataset(tf.estimator.ModeKeys.EVAL, args),
            train_steps = train_steps,
            eval_steps = 1,
            min_eval_frequency = steps_in_epoch,
            export_strategies = tf.contrib.learn.utils.saved_model_export_utils.make_export_strategy(
                serving_input_fn = create_serving_input_fn(args)
            )
        )
    
    learn_runner.run(experiment_fn, args['output_dir'])