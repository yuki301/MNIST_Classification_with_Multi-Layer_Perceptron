import struct
import numpy as np
import matplotlib.pyplot as plt


def read_idx(filename):
    # 打开文件
    with open(filename, 'rb') as file:
        # 处理魔数部分
        file.read(2)  # file.read按字节读取文件
        data_type = struct.unpack('>B', file.read(1))[0]
        # struct.unpack将二进制数据解包为无符号整数，`>I`表示大端优先，B是读取1字节，I是4字节
        dimension_number = struct.unpack('>B', file.read(1))[0]

        # 处理维度大小部分
        dimension_size = []
        for _ in range(dimension_number):
            dimension_size.append(struct.unpack('>I', file.read(4))[0])

        # 处理数据部分
        data = np.frombuffer(file.read(), dtype=np.uint8)
        # np.frombuffer从缓冲区创建一个NumPy数组，file.read不带参数读取文件剩余部分，dtype为数据项类型
        data = data.reshape(dimension_size)  # np.reshape将NumPy数组按照参数规定的维度重新组织

        return data


def show_pic_from_idx(data, begin=0, number=1, labdata=None):
    # 根据参数显示图片
    for i in range(number):
        plt.imshow(data[begin+i], cmap='gray')
        if labdata is None:  # 如果传入了标签数据，就显示标签
            plt.title("Image "+str(begin+i))
        else:
            plt.title("Image "+str(begin+i)+" Label:"+str(labdata[begin+i]))
        plt.show()


data = read_idx('t10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
labdata = read_idx('t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')
show_pic_from_idx(data, 100, 10, labdata)





