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

        # ---

        self.conv7 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn7   = tf.keras.layers.BatchNormalization()

        self.conv8 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn8   = tf.keras.layers.BatchNormalization()

        self.conv9 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn9   = tf.keras.layers.BatchNormalization()

        self.conv10 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn10   = tf.keras.layers.BatchNormalization()

        self.conv11 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn11   = tf.keras.layers.BatchNormalization()

        self.conv12 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=(2,2), padding='SAME', use_bias=False, activation=tf.identity)
        self.bn12   = tf.keras.layers.BatchNormalization()



    def call(self, inputs):

        #print('input:',inputs.shape)

        # Encoder
        conv1 = tf.nn.relu(self.conv1(inputs))
        conv2 = tf.nn.relu(self.bn2(self.conv2(conv1)))
        conv3 = tf.nn.relu(self.bn3(self.conv3(conv2)))
        conv4 = tf.nn.relu(self.bn4(self.conv4(conv3)))
        conv5 = tf.nn.relu(self.bn5(self.conv5(conv4)))
        conv6 = tf.nn.relu(self.bn6(self.conv6(conv5)))

        # Decoder with skip connections
        conv7  = tf.nn.leaky_relu(self.bn7(self.conv7(conv6)))
        conv7  = self.concat([conv7, conv5])

        conv8  = tf.nn.leaky_relu(self.bn8(self.conv8(conv7)))
        conv8  = self.concat([conv8, conv4])

        conv9  = tf.nn.leaky_relu(self.bn9(self.conv9(conv8)))
        conv9  = self.concat([conv9, conv3])

        conv10 = tf.nn.leaky_relu(self.bn10(self.conv10(conv9)))
        conv10  = self.concat([conv10, conv2])

        conv11 = tf.nn.leaky_relu(self.bn11(self.conv11(conv10)))
        conv11 = self.concat([conv11, conv1])

        conv12 = tf.nn.tanh(self.bn12(self.conv12(conv11)))

        return conv12



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
