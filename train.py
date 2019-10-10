import tensorflow as tf
import numpy as np
import argparse
import time
import cv2
import os

# My imports
import pix2pix
import ops

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', required=False, default=4, type=int, help='Batch size')
    parser.add_argument('--l1_weight', required=False, default=100., type=float, help='Weight for L1 loss')
    parser.add_argument('--epochs', required=False, default=50, type=int, help='Number of epochs')
    parser.add_argument('--dataset', required=False, default='maps', type=str, help='Dataset to use')
    parser.add_argument('--save_freq', required=False, default=500, type=int, help='How often to save a model and run on eval split')
    parser.add_argument('--print_freq', required=False, default=10, type=int, help='How often to print model loss')
    parser.add_argument('--direction', required=False, default='ytox', type=str, help='Direction to generate images in')
    args = parser.parse_args()

    args.checkpoint_dir = ops.createCheckpoint(args)

    # Load up data
    train_paths = np.asarray(ops.getPaths(os.path.join('datasets',args.dataset,'train')))
    test_paths = np.asarray(ops.getPaths(os.path.join('datasets',args.dataset,'val')))

    num_train = len(train_paths)
    num_test  = len(test_paths)

    learning_rate = 0.0002
    beta_1 = 0.5
    beta_2 = 0.999

    # Save variables on the cpu
    with tf.device('/CPU:0'):
        generator = pix2pix.Pix2PixGenerator()
        discriminator = pix2pix.Pix2PixDiscriminator()

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

        train_summary_writer = tf.summary.create_file_writer(os.path.join(args.checkpoint_dir, 'logs'))

    checkpoint = tf.train.Checkpoint(
        generator=generator,
        generator_optimizer=generator_optimizer,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer
    )

    manager = tf.train.CheckpointManager(checkpoint, directory=args.checkpoint_dir, max_to_keep=1)
    checkpoint.restore(manager.latest_checkpoint)

    def getBatch(batch_paths):

        batch_images_x = np.empty((len(batch_paths), 256, 256, 3), dtype=np.float32)
        batch_images_y = np.empty((len(batch_paths), 256, 256, 3), dtype=np.float32)

        for i,img_path in enumerate(batch_paths):

            img_xy = cv2.imread(img_path)
            img_xy = cv2.resize(img_xy, (512, 256)).astype(np.float32)
            img_xy = ops.normalize(img_xy)

            img_x = img_xy[:,:256,:]
            img_y = img_xy[:,256:,:]

            if args.direction == 'ytox':
                img_x, img_y = img_y, img_x

            batch_images_x[i, ...] = img_x
            batch_images_y[i, ...] = img_y

        return tf.convert_to_tensor(batch_images_x), tf.convert_to_tensor(batch_images_y)


    @tf.function
    def trainingStepG(batch_images_x, batch_images_y):

        with tf.GradientTape() as tape:

            # Generate a batch of fake images
            batch_images_g = generator(batch_images_x)

            # Send fake images to the discriminator
            d_fake = discriminator(batch_images_x, batch_images_g)

            # GAN loss
            errG = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

            # Reconstruction loss
            err_l1 = args.l1_weight*tf.reduce_mean(tf.abs(batch_images_g-batch_images_y))

            gradients = tape.gradient(errG+err_l1, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

        return {
            'errG' : errG,
            'err_l1' : err_l1
        }

    @tf.function
    def trainingStepD(batch_images_x, batch_images_y):

        with tf.GradientTape() as tape:

            # Generate a batch of fake images
            batch_images_g = generator(batch_images_x)

            # Send fake images to the discriminator
            d_fake = discriminator(batch_images_x, batch_images_g)

            # Send real images to the discriminator
            d_real = discriminator(batch_images_x, batch_images_y)

            # Error for discriminator
            errD_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))
            errD_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_real))
            errD = tf.reduce_mean(errD_real + errD_fake)

        gradients = tape.gradient(errD, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

        return {
            'errD' : errD
        }


    def train():

        test_idx = np.random.choice(np.arange(num_test), args.batch_size, replace=False)
        batch_test_paths = test_paths[test_idx]

        step = int(generator_optimizer.iterations)
        if step == 0: step = 1

        epoch_num = int(step/(num_train/args.batch_size))

        while epoch_num < args.epochs:
        
            idx = np.random.choice(np.arange(num_train), args.batch_size, replace=False)
            batch_paths = train_paths[idx]

            batch_images_x, batch_images_y = getBatch(batch_paths)

            loss_G = trainingStepG(batch_images_x, batch_images_y)
            loss_D = trainingStepD(batch_images_x, batch_images_y)

            errD = loss_D['errD']
            errG = loss_G['errG']
            err_l1 = loss_G['err_l1']

            print(' | epoch: '+str(epoch_num)+' | step: '+str(step)+' | errD: '+str(errD.numpy())+' | errG: '+str(errG.numpy())+' | err_l1: '+str(err_l1.numpy()))

            step += 1

            epoch_num = int(step/(num_train/args.batch_size))

            if step%args.save_freq == 0:
                print('Saving model')
                manager.save()
                print('Model saved')
 
                batch_images_x, batch_images_y = getBatch(batch_test_paths)
                batch_images_g = generator(batch_images_x)

                canvas = np.empty((256*4, 256*3, 3), dtype=np.uint8)
                i = 0
                for img_x, img_g, img_y in zip(batch_images_x, batch_images_g, batch_images_y):

                    img_x = ops.unnormalize(img_x.numpy()).astype(np.uint8)
                    img_g = ops.unnormalize(img_g.numpy()).astype(np.uint8)
                    img_y = ops.unnormalize(img_y.numpy()).astype(np.uint8)

                    canvas[i*256:(i+1)*256, :256, :] = img_x
                    canvas[i*256:(i+1)*256, 256:512, :] = img_g
                    canvas[i*256:(i+1)*256, 512:, :] = img_y

                    if i == 4: break
                    i += 1
                cv2.imwrite(os.path.join(args.checkpoint_dir, 'images', 'canvas_'+str(step)+'.png'), canvas)


    with train_summary_writer.as_default():
        train()

if __name__ == '__main__':
    main()


