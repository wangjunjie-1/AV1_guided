import sys
import cv2
from numpy import *
if not hasattr(sys, 'argv'):
    sys.argv = ['']

from model.VRCNN import   base_model as model

from UTILS_normal import *
#from cnn_n import saveImg
from shutil import copyfile
#全局变量，
tplt1 = "{0:^30}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}"  #\t{4:^10}\t{5:^10}
tplt2 = "{0:^30}\t{1:^10}\t{2:^10}"

model_set = {
    #intra ttainset v24
    "VRCNN_V1_QP6~15" :r"E:\WJJ\av1_crlc\crlc_model\vrcnn\v1\qp12\VRCNN_v1_qp12_20210803_018_0.00.ckpt",     #1024 0 15 -7 8
    "VRCNN_V1_QP16~25":r"E:\WJJ\av1_crlc\crlc_model\vrcnn\v1\qp22\VRCNN_v1_qp22_20210803_016_0.00.ckpt",  #512 2 17 -3 12
    "VRCNN_V1_QP26~30":r"E:\WJJ\av1_crlc\crlc_model\vrcnn\v1\qp28\VRCNN_v1_qp28_20210803_015_0.00.ckpt",
    "VRCNN_V1_QP31~35":r"E:\WJJ\av1_crlc\crlc_model\vrcnn\v1\qp32\VRCNN_v1_qp32_20210803_096_57.84.ckpt",
    "VRCNN_V1_QP36~45":r"E:\WJJ\av1_crlc\crlc_model\vrcnn\v1\qp43\VRCNN_v1_qp43_20210803_097_119.63.ckpt",   #1024 -15 0 -10 5
    "VRCNN_V1_QP46~55":r"E:\WJJ\av1_crlc\crlc_model\vrcnn\v1\qp53\VRCNN_v1_qp53_20210803_098_258.17.ckpt",    #256 -9 6 -10 5
    "VRCNN_V1_QP56~63":r"E:\WJJ\av1_crlc\crlc_model\vrcnn\v1\qp63\VRCNN_v1_qp63_20210803_097_572.12.ckpt",    #256 -9 6 -10 5
}


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction =1
config.gpu_options.allow_growth = True
global mx
global cnn17, cnn27, cnn37, cnn47, cnn7, cnn57


def prepare_test_data(fileOrDir):
    original_ycbcr = []
    imgCbCr = []
    gt_y = []
    fileName_list = []
    # The input is a single file.
    if type(fileOrDir) is str:
        fileName_list.append(fileOrDir)

        # w, h = getWH(fileOrDir)
        # imgY = getYdata(fileOrDir, [w, h])
        imgY = c_getYdata(fileOrDir)
        imgY = normalize(imgY)

        imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
        original_ycbcr.append([imgY, imgCbCr])

    ##The input is one directory of test images.
    elif len(fileOrDir) == 1:
        fileName_list = load_file_list(fileOrDir)
        for path in fileName_list:
            # w, h = getWH(path)
            # imgY = getYdata(path, [w, h])
            imgY = c_getYdata(path)
            imgY = normalize(imgY)

            imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
            original_ycbcr.append([imgY, imgCbCr])

    ##The input is two directories, including ground truth.
    elif len(fileOrDir) == 2:

        fileName_list = load_file_list(fileOrDir[0])
        test_list = get_train_list(load_file_list(fileOrDir[0]), load_file_list(fileOrDir[1]))
        for pair in test_list:
            filesize = os.path.getsize(pair[0])
            picsize = getWH(pair[0])[0] * getWH(pair[0])[0] * 3 // 2
            numFrames = filesize // picsize
            # if numFrames ==1:
            or_imgY = c_getYdata(pair[0])
            gt_imgY = c_getYdata(pair[1])

            # normalize
            or_imgY = normalize(or_imgY)

            or_imgY = np.resize(or_imgY, (1, or_imgY.shape[0], or_imgY.shape[1], 1))
            gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1], 1))

            ## act as a placeholder
            or_imgCbCr = 0
            original_ycbcr.append([or_imgY, or_imgCbCr])
            gt_y.append(gt_imgY)
            # else:
            #     while numFrames>0:
            #         or_imgY =getOneFrameY(pair[0])
            #         gt_imgY =getOneFrameY(pair[1])
            #         # normalize
            #         or_imgY = normalize(or_imgY)
            #
            #         or_imgY = np.resize(or_imgY, (1, or_imgY.shape[0], or_imgY.shape[1], 1))
            #         gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1], 1))
            #
            #         ## act as a placeholder
            #         or_imgCbCr = 0
            #         original_ycbcr.append([or_imgY, or_imgCbCr])
            #         gt_y.append(gt_imgY)
    else:
        print("Invalid Inputs.")
        exit(0)

    return original_ycbcr, gt_y, fileName_list


class Predict:
    input_tensor = None
    output_tensor = None
    model = None
    r = None
    gt = None
    R = None
    R_out = None

    def __init__(self, model, modelpath):
        self.graph = tf.Graph()  # 为每个类(实例)单独创建一个graph
        self.model = model
        with self.graph.as_default():
            '''
            self.input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
            #self.gt = tf.placeholder(tf.float32, shape=(1, None, None, 1))
            # self.r=tf.reshape(tf.subtract(self.gt,self.input_tensor),[1,tf.shape(self.input_tensor)[1]*tf.shape(self.input_tensor)[2],1])
            self.R = tf.make_template('shared_model', self.model)(self.input_tensor)
            # self.R = self.model(self.input_tensor)
            # self.input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
            # self.output_tensor = tf.make_template('input_scope', self.model)(self.input_tensor)
            # self.output_tensor = tf.make_template('shared_model', self.model)(self.input_tensor)
            # self.output_tensor = self.model(self.input_tensor)
            #self.output_tensor = model(self.input_tensor)
            self.output_tensor = tf.clip_by_value(self.output_tensor, 0., 1.)
            self.output_tensor = tf.multiply(self.output_tensor, 255)
            self.saver = tf.train.Saver()
            '''
            self.input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
            # self.output_tensor = tf.make_template('input_scope', self.model)(self.input_tensor)
            self.output_tensor = self.model(self.input_tensor)
            self.output_tensor = tf.clip_by_value(self.output_tensor, 0, 1)
            self.output_tensor = tf.multiply(self.output_tensor, 255)
            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=self.graph, config=config)  # 创建新的sess
        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())
                self.saver.restore(self.sess, modelpath)  # 从恢复点恢复参数
                print(modelpath)
    def predict(self, fileOrDir):
        # print("------------")
        if (isinstance(fileOrDir, str)):
            original_ycbcr, gt_y, fileName_list = prepare_test_data(fileOrDir)
            imgY = original_ycbcr[0][0]

        elif type(fileOrDir) is np.ndarray:
            imgY = fileOrDir
            #imgY = normalize(np.reshape(fileOrDir, (1, len(fileOrDir), len(fileOrDir[0]), 1)))

        elif (isinstance(fileOrDir, list)):
            fileOrDir = np.asarray(fileOrDir, dtype='float32')
            # fileOrDir = fileOrDir / 255
            imgY = normalize(np.reshape(fileOrDir, (1, len(fileOrDir), len(fileOrDir[0]), 1)))

        else:
            imgY=None

        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: imgY})
                out = np.reshape(out, (out.shape[1], out.shape[2]))
                out = np.around(out)
                out = out.astype('int')
                out = out.tolist()
                return out
# class Predict:
#     input_tensor = None
#     a = None
#     output_tensor = None
#     model = None
#     r = None
#     gt = None
#     R = None
#     R_out = None
#
#     def __init__(self, model, modelpath):
#         self.graph = tf.Graph()  # 为每个类(实例)单独创建一个graph
#         self.model = model
#         with self.graph.as_default():
#             self.input_tensor= tf.placeholder(tf.float32, shape=(1, None, None, 1))
#             # self.output_tensor = tf.make_template('input_scope', self.model)(self.input_tensor)
#             self.output_tensor , self.a= tf.make_template('shared_model', self.model)(self.input_tensor)
#             self.output_tensor = tf.clip_by_value(self.output_tensor, 0, 1)
#             self.output_tensor = tf.multiply(self.output_tensor, 255)
#             self.saver = tf.train.Saver()
#
#         self.sess = tf.Session(graph=self.graph, config=config)  # 创建新的sess
#         with self.sess.as_default():
#             with self.graph.as_default():
#                 self.sess.run(tf.global_variables_initializer())
#                 self.saver.restore(self.sess, modelpath)  # 从恢复点恢复参数
#                 print(modelpath)
#     def predict(self, fileOrDir):
#         # print("------------")
#         if (isinstance(fileOrDir, str)):
#             original_ycbcr, gt_y, fileName_list = prepare_test_data(fileOrDir)
#             imgY = original_ycbcr[0][0]
#
#         elif type(fileOrDir) is np.ndarray:
#             imgY = fileOrDir
#             #imgY = normalize(np.reshape(fileOrDir, (1, len(fileOrDir), len(fileOrDir[0]), 1)))
#
#         elif (isinstance(fileOrDir, list)):
#             fileOrDir = np.asarray(fileOrDir, dtype='float32')
#             # fileOrDir = fileOrDir / 255
#             imgY = normalize(np.reshape(fileOrDir, (1, len(fileOrDir), len(fileOrDir[0]), 1)))
#
#         else:
#             imgY=None
#
#         with self.sess.as_default():
#             with self.sess.graph.as_default():
#                 a = self.sess.run(self.a,feed_dict={self.input_tensor: imgY})
#                 print(a)
#                 out= self.sess.run(self.output_tensor,feed_dict={self.input_tensor: imgY})
#                 out = np.reshape(out, (out.shape[1], out.shape[2]))
#                 out = np.around(out)
#                 out = out.astype('int')
#                 out = out.tolist()
#                 return out

def init(sliceType, QP):
    global model, cnn12,cnn22,cnn28,cnn33,cnn43,cnn53,cnn63
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxinit++++++++++++++++++++++++++++++++++++++")
    cnn12 = Predict(model, model_set["VRCNN_V1_QP6~15"])
    cnn22 = Predict(model, model_set["VRCNN_V1_QP16~25"])
    cnn28 = Predict(model, model_set["VRCNN_V1_QP26~30"])
    cnn33 = Predict(model, model_set["VRCNN_V1_QP31~35"])
    cnn43 = Predict(model, model_set["VRCNN_V1_QP36~45"])
    cnn53 = Predict(model, model_set["VRCNN_V1_QP46~55"])
    cnn63 = Predict(model, model_set["VRCNN_V1_QP56~63"])
    print("ffffffffffffffffffffffffffffffffffffffffffffffffffff")


def predict(dgr, QP,frametype):
    global model, cnn12, cnn22, cnn28, cnn33, cnn43, cnn53, cnn63
    QP = QP / 4
    Arange = 16
    print("qp in python is ")
    print(QP)

    if QP < 17:
        R = cnn12.predict(dgr)
    elif 17 <= QP < 27:
        R = cnn22.predict(dgr)
    elif 27 <= QP < 30:
        R = cnn28.predict(dgr)
    elif 30 <= QP < 37:
        R = cnn33.predict(dgr)
    elif 37 <= QP < 47:
        R = cnn43.predict(dgr)
    elif 47 <= QP < 57:
        R = cnn53.predict(dgr)
    else:
        R = cnn63.predict(dgr)
    #R = cnn53.predict(dgr)
    print("return rec")

    return R

def showImg(inp):

    h, w = inp[0], inp[1]
    tem = np.asarray(inp, dtype='uint8')
    #np.save(r"H:\KONG\cnn_2K%f" % time.time(),tem)
    tem = Image.fromarray(tem, 'L')
    tem.show()
    #tem.save("D:/rec/FromPython%f.jpg" % time.time())

def test_all_ckpt(modelPath):
    low_img = r"E:\WJJ\av1_crlc\cnn_train\test_set\QP63"
    heigh_img = r"E:\WJJ\av1_crlc\cnn_train\test_set\label"

    #low_img = r"E:\zhen\av2\10bit_frame\highbit_frame_qp53"
    #heigh_img = r"E:\zhen\av2\10bit_frame\highbit_frame_lable"
    NUM_CNN=1 #cnn 次数
    original_ycbcr, gt_y, fileName_list = prepare_test_data([low_img,heigh_img])
    #print(original_ycbcr, gt_y, fileName_list)
    total_imgs = len(fileName_list)

    tem = [f for f in os.listdir(modelPath) if 'data' in f]
    ckptFiles = sorted([r.split('.data')[0] for r in tem])
    max_psnr = 0
    max_epoch = 0
    max_ckpt_psnr = 0


    times = 0


    for ckpt in ckptFiles:
        cur_ckpt_psnr=0
        #epoch = int(ckpt.split('_')[3])
        # print(ckpt.split('.')[0].split('_'))
        epoch = int(ckpt.split('.')[0].split('_')[-2])
        # loss =int(ckpt.split('.')[0].split('_')[-1])
        #
        # if  epoch  <400:
        #    continue
        #print(epoch)
        print(os.path.join(modelPath, ckpt))

        predictor = Predict(model, os.path.join(modelPath, ckpt))


        img_index = [14, 17, 4, 2, 7, 10, 12, 3, 0, 13, 16, 5, 6, 1, 15, 8, 9, 11]
        #img_index = [20, 24, 5, 3, 8, 15, 18, 4, 1, 19, 22, 6, 7, 2, 21, 11, 14, 16, 10, 17, 23, 9, 12, 25, 26, 0, 13]

        for i in img_index:
            # if i>5:
            #    continue
            imgY = original_ycbcr[i][0]
            gtY = gt_y[i] if gt_y else 0


            #showImg(rec)
            #print(np.shape(np.reshape(imgY, [np.shape(imgY)[1],np.shape(imgY)[2]])))
            #cur_psnr[cnnTime]=psnr(denormalize(np.reshape(imgY, [np.shape(imgY)[1],np.shape(imgY)[2]])),np.reshape(gtY, [np.shape(imgY)[1],np.shape(imgY)[2]]))
            cur_psnr=[]
            time_start = time.time()
            rec = predictor.predict(imgY)
            time_end = time.time()
            cost_time = time_end - time_start
            times = times+cost_time
            cur_psnr.append(psnr(rec,np.reshape(gtY, np.shape(rec))))
            # for cc in range(2,NUM_CNN+1):
            #     rec = predictor.predict(rec)
            #     cur_psnr.append(psnr(rec, np.reshape(gtY, np.shape(rec))))

            # print(cur_psnr)


            #print(len(cur_psnr))
            cur_ckpt_psnr=cur_ckpt_psnr+np.mean(cur_psnr)
            # print(tplt2.format(os.path.basename(fileName_list[i]), cur_psnr,psnr(denormalize(np.reshape(imgY, np.shape(rec))),np.reshape(gtY, np.shape(rec)))))
            print("%30s"%os.path.basename(fileName_list[i]),end="")
            for cc in cur_psnr:
                print("       %.5f"%cc,end="")
            print("       %.5f" % np.mean(cur_psnr), end="")
            print("       %.5f"%psnr(denormalize(np.reshape(imgY, np.shape(rec))),np.reshape(gtY, np.shape(rec))))
        print("cost_time:", times / 18, "s")
        if(cur_ckpt_psnr/total_imgs>max_ckpt_psnr):
            max_ckpt_psnr=cur_ckpt_psnr/total_imgs
            max_epoch=epoch
        print("______________________________________________________________")
        print(epoch,cur_ckpt_psnr/total_imgs,max_epoch,max_ckpt_psnr)


if __name__ == '__main__':


     #test_all_ckpt(r"E:\WJJ\aom\aom_build\Release\checkpoints\VRCNN_v1_qp32_20210803")
     test_all_ckpt(r"E:\WJJ\aom\aom_build\Release\checkpoints\VRCNN_v1_qp63_20210803")
     # init(45, 654)
     # dgr = r"E:\WJJ\av1_crlc\cnn_train\test\QP32\label\BasketballDrill_832x480.yuv"
     # src = r"E:\WJJ\av1_crlc\cnn_train\test\QP32\label\BasketballDrill_832x480.yuv"
     #
     # print(predict(dgr, 128))
     # print("finish-----------------")