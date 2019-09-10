
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

"HOG特征检测：2*2细胞／区间、24*24像素／细胞、9个直方图通道,步长为1"
class Hog_description():
    def __init__(self, img, cell_size=16, bin_size=9):
        #cell_size可以依据图片大小等各方面因素来调 本次图像大，本次测试用24效果好
        self.img = img / float(np.max(img))  #归一化
        self.img = np.power(self.img, 0.8)#伽马校正  数字可调
        #self.img = self.img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert(type(self.bin_size) == int, "bin_size should be integer")
        assert(type(self.cell_size) == int, "cell_size should be integer,")
        assert(type(self.angle_unit) == int, "bin_size should be divisible by 360")

    def global_gradient(self):
        gradient_value_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # 梯度值
        gradient_value_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_value_x, 0.5, gradient_value_y, 0.5, 0)  # 梯度幅值
        gradient_angle = cv2.phase(gradient_value_x, gradient_value_y, angleInDegrees=True)  # 梯度角
        return gradient_magnitude, gradient_angle

    # 计算梯度方向直方图
    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for k in range(cell_magnitude.shape[0]):
            for i in range(cell_magnitude.shape[1]):
                grad_strength = cell_magnitude[k][i]
                grad_angle = cell_angle[k][i]
                min_angle = int(grad_angle / self.angle_unit) % self.bin_size
                max_angle = (min_angle + 1) % self.bin_size
                mod = grad_angle % self.angle_unit
                orientation_centers[min_angle] += (grad_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (grad_strength * (mod / self.angle_unit))
        return orientation_centers

    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        print(cell_gradient_vector.shape)

        # 计算每个cell的 梯度方向直方图
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size: (i + 1) * self.cell_size,
                                 j * self.cell_size: (j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size: (i + 1) * self.cell_size,
                             j * self.cell_size: (j + 1) * self.cell_size]
                # print(cell_angle.max())
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        # 统计Block的梯度信息
        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])

                # block内归一化，可以消除局部光照，前景背景对比度的变换
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if (magnitude != 0):
                    normalize = lambda block_vector, magnitude: [e / magnitude for e in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        print(np.array(hog_vector).shape)
        return hog_vector, hog_image

    #  可视化cell梯度直方图
    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image


img = cv2.imread("kobe.jpg", cv2.IMREAD_GRAYSCALE)
hog = Hog_description(img, cell_size=24, bin_size=9)
vector, image = hog.extract()
plt.imshow(image, cmap=plt.cm.gray)
plt.show()

cv2.imshow('Image', img)
cv2.waitKey(0)















