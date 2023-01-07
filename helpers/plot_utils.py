from matplotlib import pyplot as plt
from globals import RESOURCES_ROOT, BUCKET_NAME
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import itertools
import boto3 
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')
import io

def create_plot(excercise:str, history: list):
    '''
    Create a plot of the training history to see whether you're overfitting.
    args:
        excercise: the excercise name
        history: the history object returned by the model.fit() function'''
    # Visualize the training history to see whether you're overfitting.
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['TRAIN', 'VAL'], loc='lower right')
    plt.show()
    #Save the matplot figure
    image_path = f'resources/{excercise}/plots/{excercise}_model_accuracy.png'
    plt.savefig(f'{excercise}_model_accuracy.png')

    # Create a binary stream in memory
    buf = io.BytesIO()

    # Save the plot to the stream
    plt.savefig(buf, format='png')

    # Reset the stream position to the beginning of the stream
    buf.seek(0)

    # Save the plot to S3
    s3_client.put_object(Bucket=BUCKET_NAME, Key=image_path, Body=buf)
    plt.close()    

def plot_confusion_matrix(cm: np.ndarray , classes: list, excercise:str,   normalize = False ,cmap=plt.cm.Blues):
  """Plots the confusion matrix
  Args:
    cm: the confusion matrix
    classes: the class names
    excercise: the excercise name
    cmap: the color map
    """
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')
  title = f'Confusion matrix for {excercise} model'
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=55)
  plt.yticks(tick_marks, classes)
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
  plt.show()
  image_path = f'resources/{excercise}/plots/{excercise}_cm.png'
  plt.savefig(f'{excercise}_cm.png')

  # Create a binary stream in memory
  buf = io.BytesIO()

  # Save the plot to the stream
  plt.savefig(buf, format='png')

  # Reset the stream position to the beginning of the stream
  buf.seek(0)

  # Save the plot to S3
  s3_client.put_object(Bucket=BUCKET_NAME, Key=image_path, Body=buf)
  plt.close()    
