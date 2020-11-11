import io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds 


def data_batches_get():
    (train,test) , info =tfds.load("imdb_reviews/subwords8k",
                                    split =(tfds.Split.TRAIN,tfds.Split.TEST),
                                    with_info =True,as_supervised=True)


    encoder  = info.features["text"].encoder

    padding_shape = ([None],())

    train_batch =train.shuffle(1000).padded_batch(10,padded_shapes = padding_shape)

    test_batch  = test.shuffle(1000).padded_batch(10,padded_shapes = padding_shape)

    return train_batch,test_batch,encoder

def model_get(encoder,embedding_dim = 16):
    
    #Stack of layers
    model=keras.Sequential([layers.Embedding(encoder.vocab_size,embedding_dim),
                            layers.GlobalAveragePooling1D(),
                            layers.Dense(1,activation = "sigmoid")])


    model.compile(optimizer = "adam" , loss = "binary_crossentropy",metrics = ["accuracy"])
    
    return model


def plot(history):
    hist_dict =hist.history
    acc = hist_dict["accuracy"]

    val_acc = hist_dict["val_accuracy"]
    epochs = range(1,len(acc)+1)

    plt.figure(figsize = (12,9))
    plt.plot(epochs,acc,"b",label = "Training accuracy")
    plt.plot(epochs,val_acc , "r", label = "Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.show()

def embeddings_redeem(model,encoder):
    vectors_out = io.open("vecs.tsv", "w", encoding="utf-8")
    metadata_out = io.open("meta.tsv",  "w", encoding = "utf-8")
    weights = model.layers[0].get_weights()[0]

    for num,word in enumerate(encoder.subwords):
        vector = weights[num+1]
        metadata_out.write(word + "\n")
        vectors_out.write("\t".join([str(x) for x in vector]) + "\n")
    
    vectors_out.close()
    metadata_out.close()
    
    


train_batches,test_batches,encoder  = data_batches_get()
model =model_get(encoder)
hist = model.fit(train_batches,epochs=10 , validation_data = test_batches,
                    validation_steps=20)

plot(hist)
embeddings_redeem(model,encoder)