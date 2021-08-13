import os, glob, re, sys, argparse, time, random
from random import shuffle
import tensorflow as tf
import numpy as np
from PIL import Image
# from MODEL import model
# from cnn_models_class import *
# from cnn_models_class import Predict
from model.VRCNN import base_model as  model
from UTILS_vrcnn import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EXP_DATA = 'VRCNN_v1_qp53_20210812'  # naming model
LOW_DATA_PATH = r"E:\0WZY\Data_Set\Train_Set\av1_deblock_nocdefLr"  # The path where data is stored
HIGH_DATA_PATH = r"E:\0WZY\Data_Set\Train_Set\div2k_train_hr_yuv"  # The path where label is stored

#LOW_DATA_PATH1 = r"E:\zhen\av1_ori\dataset\qp53\img_qp53"      #相对高质量帧
#HIGH_DATA_PATH1 = r"E:\zhen\av1_ori\div2k_ori"                 #高质量帧


LOG_PATH = "./logs/%s/" % (EXP_DATA)
CKPT_PATH = "./checkpoints/%s/" % (EXP_DATA)  # Store the trained models
SAMPLE_PATH = "./samples/%s/" % (EXP_DATA)  # Store result pic
PATCH_SIZE = (64, 64)  # 输入网络图像的每块的大小
BATCH_SIZE = 64  # 一张图像分为64块
BASE_LR = 1e-4  # Base learning rate
LR_DECAY_RATE = 0.5
LR_DECAY_STEP = 20
MAX_EPOCH = 400
QP_MIN=46
QP_MAX=55


parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
#model_path = r"E:\WJJ\aom\aom_build\Release\checkpoints\VRCNN_v1_qp63_20210803\VRCNN_v1_qp63_20210803_023_571.11.ckpt" #qp43 restore point
model_path =""



if __name__ == '__main__':
    start = time.time()
    train_list = get_train_list(load_file_list(LOW_DATA_PATH), load_file_list(HIGH_DATA_PATH),QP_MIN,QP_MAX)
#    train_listcnn23 = get_train_list(load_file_list(LOW_DATA_PATH1), load_file_list(HIGH_DATA_PATH1))


    train_input = tf.compat.v1.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    train_gt = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))


    train_output = model(train_input)
    train_output = tf.clip_by_value(train_output, 0., 1.)
#       train_input_cnn23 = tf.clip_by_value(train_input_cnn23, 0., 1.)
    with tf.name_scope('loss_scope'), tf.device("/gpu:0"):

        loss = tf.reduce_sum(tf.square(tf.subtract(train_output, train_gt))) #  L2   loss函数
        #loss = tf.reduce_sum(tf.abs(tf.subtract(train_output, train_gt)))    #  L1   loss函数
        #loss = loss1

        # loss = tf.reduce_mean(- tf.reduce_mean(train_gt * tf.log(train_output), reduction_indices=[1]))


        weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
        # print(weights)
        loss += tf.add_n(weights) * 1e-4
        #在计算图中占位
        avg_loss = tf.placeholder('float32')
        tf.compat.v1.summary.scalar("avg_loss", avg_loss)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(BASE_LR, global_step, LR_DECAY_STEP * 1000, LR_DECAY_RATE,
                                               staircase=True)
    print(learning_rate)
    tf.summary.scalar("learning rate", learning_rate)

    optimizer_adam = tf.compat.v1.train.AdamOptimizer(learning_rate, 0.9)
    opt_adam = optimizer_adam.minimize(loss, global_step=global_step)

    #optimizer_SGD = tf.train.GradientDescentOptimizer(learning_rate)
    #opt_SGD = optimizer_SGD.minimize(loss, global_step=global_step)
    opt_SGD = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(loss, global_step=global_step)

    saver = tf.compat.v1.train.Saver(max_to_keep=0)

    # with tf.name_scope('testInput_scope'):
    #     test_input = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    #     test_output, _ = shared_model(test_input)
    #     test_input_data, test_gt_data, test_cbcr_data = prepare_nn_data(train_list, 56)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8

    with tf.Session(config=config) as sess:
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        if not os.path.exists(os.path.dirname(CKPT_PATH)):
            os.makedirs(os.path.dirname(CKPT_PATH))
        if not os.path.exists(SAMPLE_PATH):
            os.makedirs(SAMPLE_PATH)

        merged = tf.compat.v1.summary.merge_all()
        file_writer =  tf.compat.v1.summary.FileWriter(LOG_PATH, sess.graph)

        sess.run(tf.global_variables_initializer())
        last_epoch=0

        if model_path:
            print("restore model...",model_path)
            saver.restore(sess, model_path)
            last_epoch =int(os.path.basename(model_path).split("_")[-2])
            print(last_epoch)
        print("prepare_time:", time.time() - start)

        for epoch in range(last_epoch,last_epoch+MAX_EPOCH):

            shuffle(train_list)

            total_g_loss, n_iter = 0, 0

            epoch_time = time.time()
            total_get_data_time, total_network_time = 0, 0
            for idx in range(1000):
                get_data_time = time.time()
                input_data, gt_data = prepare_nn_data(train_list)
                #temp = total_get_data_time
                total_get_data_time += (time.time() - get_data_time)
                #print(idx, total_get_data_time - temp)
                network_time = time.time()
                feed_dict = {train_input: input_data, train_gt: gt_data}

                _, l, output, g_step = sess.run([opt_adam, loss, train_output, global_step], feed_dict=feed_dict)
                total_network_time += (time.time() - network_time)
                total_g_loss += l
                n_iter += 1
                # print(output[0])
                # file_writer.add_summary(summary, g_step)
                del input_data, gt_data, output
            lr, summary = sess.run([learning_rate, merged], {avg_loss: total_g_loss / n_iter})
            file_writer.add_summary(summary, epoch)
            # print("Epoch: [%4d/%4d]  time: %4.4f\tloss: %.8f\tlr: %.8f"%(epoch, MAX_EPOCH, time.time()-epoch_time, total_g_loss/n_iter, lr))
            tf.logging.warning(
                "Epoch: [%4d/%4d]  time: %4.4f\tloss: %.8f\tlr: %.8f\ttotal_get_data_time: %8f\ttotal_network_time: %8f" % (
                epoch, last_epoch+MAX_EPOCH, time.time() - epoch_time, total_g_loss / n_iter, lr, total_get_data_time,
                total_network_time))

            # test_out = sess.run(test_output, feed_dict={test_input: test_input_data})
            # test_out = denormalize(test_out)
            # save_images(test_out, test_cbcr_data, [8, 8], os.path.join(SAMPLE_PATH,"epoch%s.png"%epoch))
            #if ((epoch + 1) % 10 == 0):
            saver.save(sess, os.path.join(CKPT_PATH, "%s_%03d_%.2f.ckpt" % (EXP_DATA,epoch, total_g_loss / n_iter)))
