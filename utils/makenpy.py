import numpy as np
import os
from PIL import Image
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def getIITDFileArr(dir,name):
    result_arr=[]
    label_list=[]
    # 文件与标签的映射
    map={}
    # 文件与归一化后文件的映射
    map_file_result={}
    # 文件与标签序号的映射
    map_file_label={}
    # 原始标签与计数的映射
    map_new={}
    count_label=0
    # 文件计数器
    count=0
    # 取到文件夹下文件名
    file_list = os.listdir(dir)
    for file in file_list:
        # 该文件路径
        file_path = os.path.join(dir, file)
        # 针对IITD Palmprint V1
        # 取标签【此处需要根据数据文件名变化】
        label=file.split(".")[0].split("_")[0]
        # 文件与原始标签映射 存入字典
        map[file]=label
        # 如果是新标签，进行标签更新
        if label not in label_list:
            label_list.append(label)
            # 对新的原始标签进行计数，记录到字典
            map_new[label]=count_label
            count_label=count_label+1
        img=Image.open(file_path)
        # result=np.array([])
        result=np.array([],dtype=np.float32) # 修改保存类
        if img.mode == "RGB":
            r,g,b=img.split()
            r_arr=np.array(r).reshape(img.size[0]*img.size[1])
            g_arr=np.array(g).reshape(img.size[0]*img.size[1])
            b_arr=np.array(b).reshape(img.size[0]*img.size[1])
            # 按通道进行拼接
            img_arr=np.concatenate((r_arr,g_arr,b_arr))
            result=np.concatenate((result,img_arr))
            result=result.reshape((img.size[0], img.size[1],3))
        else:
            img_arr=np.array(img).reshape(img.size[0]*img.size[1])
            result=np.concatenate((result,img_arr))
            result = result.reshape((img.size[0], img.size[1], 1))
        # 归一化
        result=result/255.0
        # 映射文件与归一化后文件
        map_file_result[file] = result
        # result_arr.append(result)
        count = count + 1

    for file in file_list:
        # 取出该文件所属标签的序号，保存到文件标签的映射字典中
        map_file_label[file] = map_new[map[file]]
        # map[file]=map_new[map[file]]
    ret_arr=[]
    # 存文件和标签的做法
    # for file in file_list:
    #     each_list=[]
    #     # 标签计数器规模的矩阵
    #     label_one_zero = np.zeros(count_label)
    #     # 取出标准化后结果
    #     result = map_file_result[file]
    #     # 取出标签
    #     label = map_file_label[file]
    #     # 标记记为1
    #     label_one_zero[label] = 1.0
    #     each_list.append(result)
    #     each_list.append(label_one_zero)
    #     ret_arr.append(each_list)

    # 将同类型文件存成一个列表的做法
    for label_i in range(count_label):
        ret_arr.append([])
    for file in file_list:
        # 取出标准化后结果
        label = map_file_label[file]
        result = map_file_result[file]
        ret_arr[label].append(result)
    # ret_arr = np.array(ret_arr, dtype=int)
    ret_arr = np.array(ret_arr)
    np.save('./datasets/'+name+".npy", ret_arr)
def getTongjiFileArr(dir,name):
    label_list=[]
    # 文件与标签的映射
    map={}
    # 文件与归一化后文件的映射
    map_file_result={}
    # 文件与标签序号的映射
    map_file_label={}
    # 原始标签与计数的映射
    map_new={}
    count_label=0
    # 文件计数器
    count=0
    # 取到文件夹下文件名
    file_list = os.listdir(dir)
    for file in file_list:
        # 该文件路径
        file_path = os.path.join(dir, file)
        # 取标签【此处需要根据数据文件名变化】
        label=str((int(file.split(".")[0])-1)//10)
        # 文件与原始标签映射 存入字典
        map[file]=label
        # 如果是新标签，进行标签更新
        if label not in label_list:
            label_list.append(label)
            # 对新的原始标签进行计数，记录到字典
            map_new[label]=count_label
            count_label=count_label+1
        img=Image.open(file_path)
        result=np.array([],dtype=np.float32) # 修改保存类
        img_arr=np.array(img).reshape(img.size[0]*img.size[1])
        result=np.concatenate((result,img_arr))
        result = result.reshape((img.size[0], img.size[1], 1))
        # 归一化
        result=result/255.0
        # 映射文件与归一化后文件
        map_file_result[file] = result
        # result_arr.append(result)
        count = count + 1

    for file in file_list:
        # 取出该文件所属标签的序号，保存到文件标签的映射字典中
        map_file_label[file] = map_new[map[file]]
    ret_arr=[]
    # 将同类型文件存成一个列表的做法
    for label_i in range(count_label):
        ret_arr.append([])
    for file in file_list:
        # 取出标准化后结果
        label = map_file_label[file]
        result = map_file_result[file]
        ret_arr[label].append(result)
    # ret_arr = np.array(ret_arr, dtype=int)
    ret_arr = np.array(ret_arr)
    np.save('F:\jupyter_notebook\DAGAN\datasets/'+name+".npy", ret_arr)
def getPolyUROIFileArr(dir,name):
    label_list=[]
    # 文件与标签的映射
    map={}
    # 文件与归一化后文件的映射
    map_file_result={}
    # 文件与标签序号的映射
    map_file_label={}
    # 原始标签与计数的映射
    map_new={}
    count_label=0
    # 文件计数器
    count=0
    # 取到文件夹下文件名
    file_list = os.listdir(dir)
    for file in file_list:
        # 该文件路径
        file_path = os.path.join(dir, file)
        # 取标签【此处需要根据数据文件名变化】
        label=(file.split("_")[1])
        # 文件与原始标签映射 存入字典
        map[file]=label
        # 如果是新标签，进行标签更新
        if label not in label_list:
            label_list.append(label)
            # 对新的原始标签进行计数，记录到字典
            map_new[label]=count_label
            count_label=count_label+1
        img=Image.open(file_path)
        result=np.array([],dtype=np.float32) # 修改保存类
        img_arr=np.array(img).reshape(img.size[0]*img.size[1])
        result=np.concatenate((result,img_arr))
        result = result.reshape((img.size[0], img.size[1], 1))
        # 归一化
        # result=result/255.0
        # 映射文件与归一化后文件
        map_file_result[file] = result
        # result_arr.append(result)
        count = count + 1

    for file in file_list:
        # 取出该文件所属标签的序号，保存到文件标签的映射字典中
        map_file_label[file] = map_new[map[file]]
    ret_arr=[]
    # 将同类型文件存成一个列表的做法
    for label_i in range(count_label):
        ret_arr.append([])
    for file in file_list:
        # 取出标准化后结果
        label = map_file_label[file]
        result = map_file_result[file]
        ret_arr[label].append(result)
    # ret_arr = np.array(ret_arr, dtype=int)
    ret_arr = np.array(ret_arr)
    print(ret_arr.shape)
    np.save('F:\jupyter_notebook\DAGAN\datasets/'+name+".npy", ret_arr)
if __name__ == '__main__':
    # 单通道
    # 德里（230，6，150，150，1）
    # dir = "E:\Documents\Matlab_work\DataBase\IITD Palmprint V1\Segmented\Left"
    # dir = "E:\Documents\Matlab_work\DataBase\IITD Palmprint V1\Segmented\Right"
    # name="IITDdata_right"
    # getIITDFileArr(dir,name)

    # 同济（600，10）
    # dir = "E:\Documents\Matlab_work\DataBase\Tongji_ROI\session2"
    # name="Tongji_session2"
    # getTongjiFileArr(dir,name)

    # 香港理工（100，6）
    dir = "E:\Documents\Matlab_work\DataBase\PolyU\PolyUROI"
    name="PolyUROI"
    getPolyUROIFileArr(dir,name)
    # 彩色
    # dir = "E:\Documents\Matlab_work\DataBase\COEP"

    # img = Image.open("E:\Documents\Matlab_work\DataBase\Tongji_ROI\session1/00001.bmp")
    # print(img.mode)