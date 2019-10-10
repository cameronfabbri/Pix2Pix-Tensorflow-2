import tensorflow as tf
from tqdm import tqdm
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
    parser.add_argument('--gan_loss', required=False, default='gan', type=str, help='Loss to use for GAN', choices=['gan','wgan'])
    parser.add_argument('--l1_weight', required=False, default=100., type=float, help='Weight for L1 loss')
    parser.add_argument('--epochs', required=False, default=50, type=int, help='Number of epochs')
    parser.add_argument('--dataset', required=False, default='KITTI', type=str, help='Dataset to use')
    parser.add_argument('--save_freq', required=False, default=500, type=int, help='How often to save a model and run on eval split')
    parser.add_argument('--print_freq', required=False, default=10, type=int, help='How often to print model loss')
    parser.add_argument('--data_dir', required=False, default='datasets', type=str, help='Root directory for data')
    args = parser.parse_args()

    checkpoint_dir = ops.createCheckpoint(args)

    # Load up data

    num_ids = len(train_ids)
    idx = np.asarray([0])
    batch_ids = train_ids[idx]
    for i,id_ in enumerate(batch_ids):
        paths = train_pairs[id_]
        left_path, right_path = paths[0], paths[1]
        buffer_left_img = imageio.imread(left_path.replace('.jpg','.png')).astype(np.float32)
        buffer_right_img = imageio.imread(right_path.replace('.jpg','.png')).astype(np.float32)

    num_train = len(train_ids)

    if args.gan_loss == 'gan':
        learning_rate = 0.0002
        beta_1 = 0.5
        beta_2 = 0.999

    if args.gan_loss == 'wgan':
        learning_rate = 0.0001
        beta_1 = 0.5
        beta_2 = 0.9

    # Save variables on the cpu
    with tf.device('/CPU:0'):
        if args.network == 'pix2pix':
            if args.num_generators == 1:
                generator = networks.Pix2PixGenerator()
            elif args.num_generators == 2:
                generator_lr = networks.Pix2PixGenerator()
                generator_rl = networks.Pix2PixGenerator()
        elif args.network == 'resnet':
            if args.num_generators == 1:
                generator = networks.ResnetGenerator()
            elif args.num_generators == 2:
                generator_lr = networks.ResnetGenerator()
                generator_rl = networks.ResnetGenerator()

        discriminator = networks.Pix2PixDiscriminator(args.gan_loss)

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

        train_summary_writer = tf.summary.create_file_writer(os.path.join(checkpoint_dir, 'logs'))

    if args.num_generators == 1:
        checkpoint = tf.train.Checkpoint(
            generator=generator,
            generator_optimizer=generator_optimizer,
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer
        )
    elif args.num_generators == 2:
        checkpoint = tf.train.Checkpoint(
            generator_lr=generator_lr,
            generator_rl=generator_rl,
            generator_optimizer=generator_optimizer,
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer
        )

    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=1)

    checkpoint.restore(manager.latest_checkpoint)

    z = tf.zeros((args.batch_size, args.image_height, args.image_width, 1), dtype=tf.float32)
    o = tf.ones((args.batch_size, args.image_height, args.image_width, 1), dtype=tf.float32)

    concat = tf.keras.layers.Concatenate(axis=3)

    label_l2r = concat([z, o])
    label_r2l = concat([o, z])

    z_val = tf.zeros((1, args.image_height, args.image_width, 1), dtype=tf.float32)
    o_val = tf.ones((1, args.image_height, args.image_width, 1), dtype=tf.float32)
    label_l2r_val = concat([z_val, o_val])
    label_r2l_val = concat([o_val, z_val])

    def getBatch(ids, pairs, batch_size):

        batchL_images = np.empty((batch_size, args.image_height, args.image_width, 3), dtype=np.float32)
        batchR_images = np.empty((batch_size, args.image_height, args.image_width, 3), dtype=np.float32)

        num_ids = len(ids)
        idx = np.random.choice(np.arange(num_ids), batch_size, replace=False)
        batch_ids = ids[idx]

        for i,id_ in enumerate(batch_ids):
            paths = pairs[id_]
            left_path, right_path = paths[0], paths[1]

            try: left_img = imageio.imread(left_path.replace('.jpg','.png')).astype(np.float32)
            except: left_img = buffer_left_img

            left_img = tf.expand_dims(tf.convert_to_tensor(left_img), 0)
            left_img = tf.image.resize(left_img, [args.image_height, args.image_width])
            left_img = data_ops.normalize(left_img)

            try: right_img = imageio.imread(right_path.replace('.jpg','.png')).astype(np.float32)
            except: right_img = buffer_right_img

            right_img = tf.expand_dims(tf.convert_to_tensor(right_img), 0)
            right_img = tf.image.resize(right_img, [args.image_height, args.image_width])
            right_img = data_ops.normalize(right_img)

            batchL_images[i, ...] = left_img
            batchR_images[i, ...] = right_img

        return batchL_images, batchR_images


    @tf.function
    def trainingStepG(batchL_images, batchR_images):

        ssim_loss = tf.constant(0.0)
        ms_ssim_loss = tf.constant(0.0)

        with tf.GradientTape() as tape:

            if args.num_generators == 1:
                gen_input_left  = concat([batchL_images, label_l2r])
                gen_input_right = concat([batchR_images, label_r2l])
                # Generated right image given left
                gen_r = generator(gen_input_left)

                # Generated left image given right
                gen_l = generator(gen_input_right)

                # Send fake images to the discriminator
                d_fake1 = discriminator(gen_input_left, gen_r)
                d_fake2 = discriminator(gen_input_right, gen_l)
                d_fake  = (d_fake1+d_fake2)/2.0

            elif args.num_generators == 2:
                # Generated right image given left
                gen_r = generator_lr(batchL_images)

                # Generated left image given right
                gen_l = generator_rl(batchR_images)

                # Send fake images to the discriminator
                d_fake1 = discriminator(batchL_images, gen_r)
                d_fake2 = discriminator(gen_l, batchR_images)
                d_fake  = (d_fake1+d_fake2)/2.0

            # GAN loss
            if args.gan_loss == 'gan':
                errG = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

            elif args.gan_loss == 'wgan':
                errG = tf.reduce_mean(-d_fake)

            # Reconstruction loss
            err_r_l1 = args.l1_weight*tf.reduce_mean(tf.abs(gen_r-batchR_images))
            err_l_l1 = args.l1_weight*tf.reduce_mean(tf.abs(gen_l-batchL_images))

            err_l1 = (err_r_l1+err_l_l1)/2.0

            if args.ssim_weight > 0.0:
                ssim_index1 = tf.reduce_mean(tf.image.ssim(tf.cast(255*batchL_images, dtype=tf.uint8), tf.cast(255*gen_l, dtype=tf.uint8), max_val=255))
                ssim_index2 = tf.reduce_mean(tf.image.ssim(tf.cast(255*batchR_images, dtype=tf.uint8), tf.cast(255*gen_r, dtype=tf.uint8), max_val=255))

                ssim_loss = -1. * ((ssim_index1 + ssim_index2) / 2.0)
                ssim_loss = args.ssim_weight * ssim_loss

            if args.ms_ssim_weight > 0.0:
                ms_ssim_index1 = tf.reduce_mean(tf.image.ssim_multiscale(tf.cast(255*batchL_images, dtype=tf.uint8), tf.cast(255*gen_l, dtype=tf.uint8), max_val=255))
                ms_ssim_index2 = tf.reduce_mean(tf.image.ssim_multiscale(tf.cast(255*batchR_images, dtype=tf.uint8), tf.cast(255*gen_r, dtype=tf.uint8), max_val=255))

                ms_ssim_loss = -1. * ((ms_ssim_index1 + ms_ssim_index2) / 2.0)
                ms_ssim_loss = args.ms_ssim_weight * ms_ssim_loss

            total_error = errG + err_l1 + ssim_loss + ms_ssim_loss

        if args.num_generators == 1:
            gradients = tape.gradient(total_error, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
        elif args.num_generators == 2:
            generator_train_vars = generator_lr.trainable_variables+generator_rl.trainable_variables
            gradients = tape.gradient(total_error, generator_train_vars)
            generator_optimizer.apply_gradients(zip(gradients, generator_train_vars))

        return {
            'errG' : errG,
            'err_l1' : err_l1,
            'err_ssim' : ssim_loss,
            'err_ms_ssim' : ms_ssim_loss,
            'total_error' : total_error,
        }

    @tf.function
    def trainingStepD(batchL_images, batchR_images):

        with tf.GradientTape() as tape:

            if args.num_generators == 1:
                gen_input_left  = concat([batchL_images, label_l2r])
                gen_input_right = concat([batchR_images, label_r2l])

                # Generated right image given left
                gen_r = generator(gen_input_left)

                # Generated left image given right
                gen_l = generator(gen_input_right)

                # Send fake images to the discriminator
                d_fake1 = discriminator(gen_input_left, gen_r)
                d_fake2 = discriminator(gen_input_right, gen_l)
                d_fake  = (d_fake1+d_fake2)/2.0

                # Send real images to the discriminator
                d_real = discriminator(batchL_images, batchR_images)

            elif args.num_generators == 2:

                # Generated right image given left
                gen_r = generator_lr(batchL_images)

                # Generated left image given right
                gen_l = generator_rl(batchR_images)

                # Send fake images to the discriminator
                d_fake1 = discriminator(batchL_images, gen_r)
                d_fake2 = discriminator(gen_l, batchR_images)
                d_fake  = (d_fake1+d_fake2)/2.0

                # Send real images to the discriminator
                d_real = discriminator(batchL_images, batchR_images)

            # Error for discriminator
            if args.gan_loss == 'gan':
                errD_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))
                errD_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_real))
                errD = tf.reduce_mean(errD_real + errD_fake)
            elif args.gan_loss == 'wgan':
                errD = tf.reduce_mean(d_fake - d_real)

                #x_hat = real_real * epsilon + (1 - epsilon) * (real_gen + gen_real)
                epsilon = tf.random.uniform([batchL_images.shape[0], 1, 1, 1], 0.0, 1.0)
                x_hat = concat([batchL_images, batchR_images]) * epsilon + (1 - epsilon) * (concat([batchL_images, gen_r]) + concat([gen_l, batchR_images]))

                with tf.GradientTape() as gt:
                    gt.watch(x_hat)
                    d_hat = discriminator(x_hat[:,:,:,:3], x_hat[:,:,:,3:])

                gt_grad = gt.gradient(d_hat, x_hat)
                ddx = tf.sqrt(tf.reduce_sum(gt_grad ** 2, axis=[1, 2]))
                d_regularizer = 10*tf.reduce_mean((ddx - 1.0) ** 2)
                errD += d_regularizer

        gradients = tape.gradient(errD, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

        return {
            'errD' : errD
        }


    def printAll(epoch, step, losses, t):

        printable = [
            ['err_l1', args.l1_weight],
            ['err_ssim', args.ssim_weight],
            ['err_ms_ssim', args.ms_ssim_weight],
        ]

        errG   = losses['errG'].numpy()
        errD   = losses['errD'].numpy()

        tf.summary.scalar('errG', losses['errG'], step=step)
        tf.summary.scalar('errD', losses['errD'], step=step)

        s = ' | epoch: '+str(epoch)
        s += '| step: '+str(step)
        s += ' | errD: %3f'%errD
        s += ' | errG: %3f'%errG

        for name, weight in printable:
            if weight > 0.0:
                s += ' | '+name+': %3f'%losses[name].numpy()
                tf.summary.scalar(name, losses[name], step=step)

        s += ' | time: %3f'%t

        print(s)

    def train():

        window_size = 3
        min_disp = 0
        num_disp = 16
        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = 3,
            P1 = 24*3*3,
            P2 = 96*3*3,
            disp12MaxDiff = 5,
            uniquenessRatio = 15,
            speckleWindowSize = 0,
            speckleRange = 0
        )
            
        if args.gan_loss == 'wgan': num_d = 5
        else: num_d = 1

        step = int(generator_optimizer.iterations)
        if step == 0: step = 1

        epoch_num = int(step/(num_train/args.batch_size))

        while epoch_num < args.epochs:
        
            epoch_num = int(step/(num_train/args.batch_size))

            batchL_images, batchR_images = getBatch(train_ids, train_pairs, args.batch_size)

            s = time.time()
            loss_G = trainingStepG(batchL_images, batchR_images)
            for i in range(num_d):
                loss_D = trainingStepD(batchL_images, batchR_images)

            printAll(epoch_num, step, {**loss_D, **loss_G}, time.time()-s)

            step += 1

            if step%args.save_freq == 0:
                print('Saving model')
                manager.save()
                print('Model saved')
                rmse_val = []

                # Loop through all val images and compute disparity
                i = 0
                canvas1 = np.empty((4*args.image_height, args.image_width, 1), dtype=np.uint8)
                canvas2 = np.empty((4*args.image_height, args.image_width, 3), dtype=np.uint8)
                for val_id in tqdm(val_ids):

                    batchL_images, batchR_images = getBatch(np.asarray([val_id]), val_pairs, 1)
                    if args.num_generators == 1:

                        # Generated right image given left
                        gen_r = generator(concat([batchL_images, label_l2r_val]))

                        # Generated left image given right
                        gen_l = generator(concat([batchR_images, label_r2l_val]))

                    elif args.num_generators == 2:
                        gen_r = generator_lr(batchL_images)

                        # Generated left image given right
                        gen_l = generator_rl(batchR_images)

                    real_l = np.squeeze(data_ops.unnormalize(batchL_images)).astype(np.uint8)
                    real_r = np.squeeze(data_ops.unnormalize(batchR_images)).astype(np.uint8)

                    gen_r = np.squeeze(data_ops.unnormalize(gen_r)).astype(np.uint8)
                    gen_l = np.squeeze(data_ops.unnormalize(gen_l)).astype(np.uint8)

                    disparityr_r, out_points, out_colors = stereo_match.stereo_image_generation(real_l, real_r, stereo)
                    disparityr_g, out_points, out_colors = stereo_match.stereo_image_generation(real_l, gen_r, stereo)
                    disparityg_r, out_points, out_colors = stereo_match.stereo_image_generation(gen_l, real_r, stereo)
                    disparityg_g, out_points, out_colors = stereo_match.stereo_image_generation(gen_l, gen_r, stereo)
                    
                    rmsr_g = sqrt(mean_squared_error(disparityr_r, disparityr_g))
                    rmsg_r = sqrt(mean_squared_error(disparityr_r, disparityg_r))
                    rmse_val.append((rmsr_g + rmsg_r) / 2.0)

                    # Only save 1 image
                    if i == 0:
                        i = 1
                        # Disparity canvas
                        canvas1[:args.image_height, :, :] = np.expand_dims(disparityr_r, 2)
                        canvas1[args.image_height:args.image_height*2, :, :] = np.expand_dims(disparityr_g, 2)
                        canvas1[args.image_height*2:args.image_height*3, :, :] = np.expand_dims(disparityg_r, 2)
                        canvas1[args.image_height*3:, :, :] = np.expand_dims(disparityg_g, 2)
                        tf.summary.image('disparity', tf.expand_dims(tf.convert_to_tensor(canvas1), 0), step)

                        # Actual images canvas
                        canvas2[:args.image_height, :, :] = np.expand_dims(real_l, 0)
                        canvas2[args.image_height:args.image_height*2, :, :] = np.expand_dims(real_r, 0)
                        canvas2[args.image_height*2:args.image_height*3, :, :] = np.expand_dims(gen_l, 0)
                        canvas2[args.image_height*3:, :, :] = np.expand_dims(gen_r, 0)
                        tf.summary.image('images', tf.expand_dims(tf.convert_to_tensor(canvas2), 0), step)

                rmse = tf.convert_to_tensor(np.mean(np.asarray(rmse_val)))

                print('val rmse:',rmse.numpy(),'\n')

                # Log RMSE to Tensorboard for val
                tf.summary.scalar('Validation RMSE', rmse, step)


    with train_summary_writer.as_default():
        train()

if __name__ == '__main__':
    main()


