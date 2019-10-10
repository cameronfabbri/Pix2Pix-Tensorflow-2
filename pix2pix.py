import tensorflow as tf

class Pix2PixGenerator(tf.keras.Model):

    def __init__(self):
        super(Pix2PixGenerator, self).__init__()

        self.concat = tf.keras.layers.Concatenate(axis=3)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(2,2), padding='SAME', use_bias=True, activation=tf.identity)

        self.conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn2   = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn3   = tf.keras.layers.BatchNormalization()

        self.conv4 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn4   = tf.keras.layers.BatchNormalization()

        self.conv5 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn5   = tf.keras.layers.BatchNormalization()
        
        self.conv6 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn6   = tf.keras.layers.BatchNormalization()

        self.conv7 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn7   = tf.keras.layers.BatchNormalization()
        
        self.conv8 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn8   = tf.keras.layers.BatchNormalization()
        
        # ---

        self.conv9 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn9   = tf.keras.layers.BatchNormalization()
        
        self.conv10 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn10   = tf.keras.layers.BatchNormalization()

        self.conv11 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn11   = tf.keras.layers.BatchNormalization()
        
        self.conv12 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn12   = tf.keras.layers.BatchNormalization()

        self.conv13 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn13   = tf.keras.layers.BatchNormalization()

        self.conv14 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn14   = tf.keras.layers.BatchNormalization()

        self.conv15 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn15   = tf.keras.layers.BatchNormalization()

        self.conv16 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn16   = tf.keras.layers.BatchNormalization()


    def call(self, inputs):

        # Encoder
        conv1 = tf.nn.leaky_relu(self.conv1(inputs))
        conv2 = tf.nn.leaky_relu(self.bn2(self.conv2(conv1)))
        conv3 = tf.nn.leaky_relu(self.bn3(self.conv3(conv2)))
        conv4 = tf.nn.leaky_relu(self.bn4(self.conv4(conv3)))
        conv5 = tf.nn.leaky_relu(self.bn5(self.conv5(conv4)))
        conv6 = tf.nn.leaky_relu(self.bn6(self.conv6(conv5)))
        conv7 = tf.nn.leaky_relu(self.bn7(self.conv7(conv6)))
        conv8 = tf.nn.leaky_relu(self.bn8(self.conv8(conv7)))

        # Decoder with skip connections
        conv9 = tf.nn.relu(self.bn9(self.conv9(conv8)))
        conv9 = self.dropout(conv9)
        conv9 = self.concat([conv9, conv7])

        conv10 = tf.nn.relu(self.bn10(self.conv10(conv9)))
        conv10 = self.dropout(conv10)
        conv10 = self.concat([conv10, conv6])

        conv11 = tf.nn.relu(self.bn11(self.conv11(conv10)))
        conv11 = self.dropout(conv11)
        conv11 = self.concat([conv11, conv5])
        
        conv12 = tf.nn.relu(self.bn12(self.conv12(conv11)))
        conv12 = self.concat([conv12, conv4])
        
        conv13 = tf.nn.relu(self.bn13(self.conv13(conv12)))
        conv13 = self.concat([conv13, conv3])
        
        conv14 = tf.nn.relu(self.bn14(self.conv14(conv13)))
        conv14 = self.concat([conv14, conv2])

        conv15 = tf.nn.relu(self.bn15(self.conv15(conv14)))
        conv15 = self.concat([conv15, conv1])
       
        conv16 = tf.nn.tanh(self.bn16(self.conv16(conv15)))

        return conv16



class Pix2PixDiscriminator(tf.keras.Model):
    """ PatchGAN Discriminator"""

    def __init__(self):
        super(Pix2PixDiscriminator, self).__init__()

        self.concat = tf.keras.layers.Concatenate(axis=3)
        self.zero_padding1 = tf.keras.layers.ZeroPadding2D()
        self.zero_padding2 = tf.keras.layers.ZeroPadding2D()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(2,2), padding='SAME',  activation=tf.identity)
        self.conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=(2,2), padding='SAME', activation=tf.identity)
        self.conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=(2,2), padding='SAME', activation=tf.identity)
        self.conv4 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=(1,1), padding='VALID', activation=tf.identity)
        self.conv5 = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=(1,1), padding='VALID', activation=tf.identity)

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()

    def call(self, input_image, output_image):
    
        inputs = self.concat([input_image, output_image])

        conv1 = tf.nn.leaky_relu(self.conv1(inputs))
        conv2 = tf.nn.leaky_relu(self.bn2(self.conv2(conv1)))

        conv3 = tf.nn.leaky_relu(self.bn3(self.conv3(conv2)))
        conv3 = self.zero_padding1(conv3)

        conv4 = tf.nn.leaky_relu(self.bn4(self.conv4(conv3)))
        conv4 = self.zero_padding2(conv4)
        
        conv5 = self.conv5(conv4)

        return conv5
