import tensorflow as tf
import tensorflow_datasets as tfds 
from tensorflow.keras import layers
from tensorflow import keras


def data_batches_get():
    (train,test) , info =tfds.load("imdb_reviews/subwords8k",
                                    split =(tfds.Split.TRAIN,tfds.Split.TEST),
                                    with_info =True,as_supervised=True)

    buffer_size = 10000
    batch_size =64

    encoder  = info.features["text"].encoder

    padding_shape = ([None],())

    train_batch =train.shuffle(buffer_size).padded_batch(batch_size,padded_shapes = padding_shape)

    test_batch  = test.shuffle(buffer_size).padded_batch(batch_size,padded_shapes = padding_shape)

    return train_batch,test_batch,encoder



def model_get(encoder,embedding_dim = 16):
    
    #Simple Model
    #model=keras.Sequential([layers.Embedding(encoder.vocab_size,embedding_dim),
     #                       layers.GlobalAveragePooling1D(),
      #                      layers.Dense(1,activation = "sigmoid")])
    
    
    #Little more complex model
   # model=keras.Sequential([layers.Embedding(encoder.vocab_size,embedding_dim),
    #                        layers.Dense(64,activation="relu"),
     #                       layers.Dropout(0.5),
      #                      layers.GlobalAveragePooling1D(),
       #                     layers.Dense(1,activation = "sigmoid")])


    #More complex model(ISSUE WHILE USING THE BIDIRECTIONAL LAYER(LSTM))(works on google colab)
    model = tf.keras.Sequential([layers.Embedding(encoder.vocab_size,64),
                             layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
                             layers.Bidirectional(tf.keras.layers.LSTM(32)),
                             layers.Dense(64,activation="relu"),
                             layers.Dropout(0.5),
                             layers.Dense(1,activation="sigmoid")]
                            )

    
    model.compile(optimizer = "adam" , loss = "binary_crossentropy",metrics = ["accuracy"])
    
    return model


train_batches,test_batches,encoder  = data_batches_get()
model =model_get(encoder)

hist = model.fit(train_batches,epochs=10 , validation_data = test_batches,
                    validation_steps=20)


def padding_to_size(vector,size):
    """
    Args:
         vector:text to pad
         size:padding size
    
    """
    zeros = [0]*(size - len(vector))
    vector.extend(zeros)
    return vector

def sample_predict(sentence , padding):
    encoded_sample_pred_text = encoder.encode(sentence)
    if padding:
        encoded_sample_pred_text = padding_to_size(encoded_sample_pred_text,64)

    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text,tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text,0))

    return predictions

def sentiment(predictions):
    if (float(predictions) >= 0.70):
        print("This is a positive review")

    elif(0.70>float(predictions) > 0.40):
        print("This is a neutral review")

    elif(float(predictions)<= 0.40):
        print("This is a negative review")

    
#Positve sample text    
sample_text = ("This movie was awesome,and the acting was incredible.Highly recommended")
predictions  = sample_predict(sample_text , padding = True) 
print("Probability this being a positive review %.2f" % predictions)
sentiment(predictions)



#Neutral sample text
sample_text = ("This movie was okay ")
predictions = sample_predict(sample_text , padding= True)
print("Probability this being a positive review %.2f" % predictions)
sentiment(predictions)


#Negative sample text
sample_text = ("This movie was too boring to even see for half an hour. ")
predictions = sample_predict(sample_text , padding= True)
print("Probability this being a positive review %.2f" % predictions)
sentiment(predictions)
