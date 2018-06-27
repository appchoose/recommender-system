import tensorflow as tf

def write_tf_records(df, by, output):
    '''Store user/item interactions in a tf.train.Example format.
    
    # Arguments
        df: A pandas dataframe listing the ratings.
        by: A character string.
        output: A file path.
    # References
        https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/core/example/example.proto
    '''
    with tf.python_io.TFRecordWriter(output) as ofp:
        for i, x in df.groupby(by):
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'key': tf.train.Feature(int64_list = tf.train.Int64List(value = [i])),
                        'indices': tf.train.Feature(int64_list = tf.train.Int64List(value = x[by].values)),
                        'values': tf.train.Feature(float_list = tf.train.FloatList(value = x['rating'].values))
                    }
                )
            )
            ofp.write(example.SerializeToString())