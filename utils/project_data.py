import torch
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)
    return data[:n], labels[:n]
def project_data(writer,dataset,classes,n=100):
    # select random images and their target indices
    dataiter = iter(dataset)
    images, labels = dataiter.next()
    
    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]
    
    # log embeddings
    features = images.view(-1, 48 * 48)
    writer.add_embedding(features,
                        metadata=class_labels)