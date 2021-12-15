import tensorflow as tf
import numpy as np


def predict_with_model(model,imgpath):
    
    image = tf.io.read_file(imgpath)
    image = tf.image.decode_png(image, channels = 3)
    image = tf.image.convert_image_dtype(image,dtype=tf.float32) 
    image = tf.image.resize(image,[60,60])
    image = tf.expand_dims(image,axis = 0)
    predictions  = model.predict(image) # [0.005,0.003,0.99,...]
    predictions = np.argmax(predictions)
    return predictions




if __name__ =="__main__":
    #img_path = "F:\\tensorflow-tutorial\\traffic_light_dataset\\German_trafficlight_dataset\\Test\\2\\00409.png"
    img_path = "F:\\tensorflow-tutorial\\traffic_light_dataset\\German_trafficlight_dataset\\Test\\11\\01257.png"
    model = tf.keras.models.load_model('./Models')
    prediction = predict_with_model(model,img_path)
    
    print(f"prediction = {prediction}")