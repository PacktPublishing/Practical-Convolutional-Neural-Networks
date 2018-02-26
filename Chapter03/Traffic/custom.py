# Execute sudo pip3 install scikit-image for the skimage
import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
import warnings
import random
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

Train_IMAGE_DIR = "/home/asif/traffic/"
Test_IMAGE_DIR = "/home/asif/traffic/"

def load_data(data_dir):
    """Loads a data set and returns two lists:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all the subdirectories of the data folder (i.e. traing or test). Each folder represents an unique label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    # Iterate for loop through the label directories and collect the data in two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".ppm")]

        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

# Load training and testing datasets.
train_data_dir = os.path.join(Train_IMAGE_DIR, "Training")
test_data_dir = os.path.join(Test_IMAGE_DIR, "Testing")

images, labels = load_data(train_data_dir)

print("Unique classes: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))

def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()

display_images_and_labels(images, labels)

def display_label_images(images, label):
    """Display images of a specific label."""
    limit = 15  # show a max of 15 images
    plt.figure(figsize=(12, 8))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 5, i)  # 3 rows, 5 per row
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

# Now let's display 10 random images
display_label_images(images, 20)

for img in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(img.shape, img.min(), img.max()))
	
# Then we resize images
images32 = [skimage.transform.resize(img, (32, 32), mode='constant') for img in images]
display_images_and_labels(images32, labels)

for img in images32[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(img.shape, img.min(), img.max()))
	
labels_array = np.array(labels)
images_array = np.array(images32)
print("labels: ", labels_array.shape, "\nimages: ", images_array.shape)

# Create the computational graph. But first let's create a graph to hold the model: 
graph = tf.Graph()
with graph.as_default():
    # Placeholders for inputs and labels.
    images_X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_X = tf.placeholder(tf.int32, [None])

    #biasInit = tf.constant_initializer(0.1, dtype=tf.float32)
    # Initializer
    biasInit = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

    # Convloution layer 1    
    conv1 = tf.contrib.layers.conv2d(images_X,  num_outputs=128,  kernel_size=[6, 6], biases_initializer=biasInit)

    # Batch normalization layer 1: can be applied as a normalizer function for conv2d and fully_connected 
    bn1 = tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=True)
    
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    pool1 = tf.contrib.layers.max_pool2d(bn1, 2, 2)

    # Convloution layer 2  
    conv2 = tf.contrib.layers.conv2d(pool1, num_outputs=256, kernel_size=[4, 4], stride=2, biases_initializer=biasInit)

    # Batch normalization layer 2: can be applied as a normalizer function for conv2d and fully_connected 
    bn2 = tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=True)
        
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    pool2 = tf.contrib.layers.max_pool2d(bn2, 2, 2)

    # Flatten the input from [None, height, width, channels] to to [None, height * width * channels] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(pool2)

    # Fully connected layer 1
    fc1 = tf.contrib.layers.fully_connected(images_flat, 512, tf.nn.relu)

    # Batch normalization layer 2: can be applied as a normalizer function for conv2d and fully_connected 
    bn3 = tf.contrib.layers.batch_norm(fc1, center=True, scale=True, is_training=True)

    # Apply dropout, if is_training is False, dropout is not applied
    fc1 = tf.layers.dropout(bn3, rate=0.25, training=True)

    # Fully connected layer 2 that generates logits of size [None, 62]. Here 62 means number of classes to be predicted.
    logits = tf.contrib.layers.fully_connected(fc1, 62, tf.nn.relu)
 
    # Convert the logits to label indexes (int) having the shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits, axis=1)

    # Define cross-entropy is the loss function, which is a good choice for classification.
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_X))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Create an optimizer, which acts as the training op.
        train = tf.train.AdamOptimizer(learning_rate=0.10).minimize(loss_op)
    
    # Finally, initizlize all the ops
    init_op = tf.global_variables_initializer()

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss_op)
print("predicted_labels: ", predicted_labels)

# Create a session to run the graph we created.
session = tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True))
session.run(init_op)	

for i in range(10):
    _, loss_value = session.run([train, loss_op], feed_dict={images_X: images_array, labels_X: labels_array})
    if i % 10 == 0:
        print("Loss: ", loss_value)
		
# Pick 10 random images
random_indexes = random.sample(range(len(images32)), 10)
random_images = [images32[i] for i in random_indexes]
random_labels = [labels[i] for i in random_indexes]

# Run the "predicted_labels" op.
predicted = session.run([predicted_labels], feed_dict={images_X: random_images})[0]
print(random_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(5, 5))
for i in range(len(random_images)):
    truth = random_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth: {0}\nPrediction:     {1}".format(truth, prediction), fontsize=12, color=color)
    plt.imshow(random_images[i])

# Load the test dataset.
test_X, test_y = load_data(test_data_dir)	

# Transform the images, just like we did with the training set.
test_images32 = [skimage.transform.resize(img, (32, 32), mode='constant') for img in test_X]
display_images_and_labels(test_images32, test_y)

# Run predictions against the full test set.
predicted = session.run([predicted_labels], feed_dict={images_X: test_images32})[0]

# Calculate how many matches we got.
match_count = sum([int(y == y_) for y, y_ in zip(test_y, predicted)])
accuracy = match_count / len(test_y)
print("Accuracy: {:.3f}".format(accuracy))

# When we're done, close the session to destroy the trained model.
session.close()
