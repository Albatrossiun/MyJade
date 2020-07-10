'''''''''''''''''''''
# @FileName:JADE.py
# @author:ZhaoXinYi
# @version:0.0.1
# @Date:2020.06.10
# @BSD
'''''''''''''''''''''
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import numpy
import matplotlib.pyplot as plt
import math
import time

class MyJADE :
    def __init__(self,
                 objectFunction, #目标函数
                 limit, # 目标函数的自变量取值范围
                 population, # 种群大小
                 generation): # 迭代次数
        self._uCR = 0.5 # 初始uCR设置为0.5
        self._uF = 0.5 # 初始uF设置为0.5
        self._dimension = len(limit) # limit的大小等同与目标函数中自变量的个数
        self._limit = limit # 把参数中指定的自变量取值范围
        self._population = population # 把参数中指定的种群大小记录到类里
        self._generation = generation # 把参数中指定的迭代次数记录到类里
        self._objectFunciton = objectFunction # 把参数中传来的目标函数记录到类里
        self._archive = [] # 初始的外部归档是一个空集
        self._minObjectFunctionValue = sys.float_info.max # 记录迭代过程中目标函数的最小值 初始值设为float可以表示的最大值
        self._bestSolution = [] # 最优解
        self._eachGenerationMin = [] # 记录每次迭代过程里的目标函数最小值
        self._FList = [] # 用来存F 动态更新uF用
        self._CRList = [] # 用来存CR 动态更新uCR用

    def __initGroup(self): # 初始化种群
        self._group = [] # 类里记录的种群 此时是一个空集
        # 根据参数中指定的种群大小  在规定的自变量取值范围内初始化种群
        for _ in range(self._population): # 外层for循环 每循环一次创建出一个个体
            body = [] # 新的个体
            for j in range(self._dimension): # 内层for循环 每循环一次，创建新个体的一个分量
                r = random.random()  # [0,1]之间的随机数
                minValue = self._limit[j][0]  # 第j个自变量取值的最小值
                maxValue = self._limit[j][1]  # 第j个自变量取值的最大值
                tmp = minValue + (maxValue - minValue) * r # minValue <= tmp <= maxValue
                body.append(tmp)
            self._group.append(body)  # 把产生的个体放到种群里

    def __sort(self, left, right):
        i = left - 1
        p = self._objectFunciton(self._group[right])
        for j in range(left, right):
            if self._objectFunciton(self._group[j]) <= p:
                i = i + 1
                self._group[i], self._group[j] = self._group[j], self._group[i]
        self._group[i+1], self._group[right] = self._group[right], self._group[i + 1]
        return i+1

    def __quickSort(self, left, right):
        if left < right:
            p = self.__sort(left, right)
            self.__quickSort(left, p - 1)
            self.__quickSort(p+1, right)

    def __sortGroup(self): # 对当前种群做排序
        self.__quickSort(0, self._population-1)

    def __bodyIsEuqal(self, body1, body2): # 判断两个个体是否相同
        for i in range(len(body1)):
            if(body1[i] != body2[i]):
                return False
        return True

    def __random(self):
        return int(random.random() * 100000000)

    def __bodyInList(self, list, body):
        for i in range(len(list)):
            if(self.__bodyIsEuqal(body, list[i])):
                return True
        return False

    def __getGroupPlusArchive(self):
        res = self._group
        for i in range(len(self._archive)):
            if not self.__bodyInList(res, self._archive[i]):
                res.append(self._archive[i])
        return res

    def __variation(self): # 变异
        # 先对当前的种群做排序
        self.__sortGroup()
        # 计算当前种群和外部归档的交集
        groupPlusArchive = self.__getGroupPlusArchive()

        index = int(len(self._group) / 2)
        bestTopBody = self._group[0:index]  # 取出前一半
        self._vlist = [] # 将类里的vlist设置为空集
        self._FList = []
        for i in range(self._population): # 外层循环 循环次数为个体数
            vi = [] # 新变异出的个体

            xi = self._group[i]  # 拿到种群中第i个个体

            x_best_index = self.__random() % len(bestTopBody)
            x_best = bestTopBody[x_best_index] # 从top中拿到x_best

            xr1 = []
            while(True): # 拿xr1 xr1不可以和xi重复
                xr1_index = self.__random() % len(self._group)
                xr1 = self._group[xr1_index]
                if not self.__bodyIsEuqal(xi, xr1): # 如果这次拿到的xr1 不等于xi
                    break

            xr2 = []
            while(True): # 拿xr2 xr2不可以和xi,xr1重复
                xr2_index = self.__random() % len(groupPlusArchive)
                xr2 = groupPlusArchive[xr2_index]
                if (not self.__bodyIsEuqal(xi, xr2)) and (not self.__bodyIsEuqal(xr1, xr2)):
                    break

            # 到这里 拿到了xi x_best xr1 xr2
            # vi = xi + F*(x_best - xi) + F*(xr1 - xr2)
            # 由于每个个体实际上有多个分量  上边的公式还可以写成
            # vi[j] = xi[j] + F*(x_best[j] - xi[j]) + F*(xr1[j] - xr2[j])   j = 0,1,...,_dimension
            # 【==注意==】 vi[j] 可能会超出自变量的取值范围 当超出范围时，需要限制

            # 计算该个体变异时的 F  F是由C(uF,0.1)的柯西分布产生的 标准柯西分布为C(0, 0.1)  C(uF,0.1) = C(0, 0.1) + uF
            # ！！！ F 如果小于0 则重新计算F F如果大于1 则令F=1
            F = -1
            while(True):
                F = numpy.random.standard_cauchy(1) + self._uF
                if F > 0:
                    break
            if F > 1:
                F = 1

            self._FList.append(F)
            for j in range(self._dimension):
                tmp = xi[j] + F*(x_best[j] - xi[j]) + F * (xr1[j] - xr2[j])
                minValue = self._limit[j][0]  # 第j个自变量取值的最小值
                maxValue = self._limit[j][1]  # 第j个自变量取值的最大值
                if tmp > maxValue or tmp < minValue:
                    r = random.random()  # [0,1]之间的随机数
                    tmp = minValue + (maxValue - minValue) * r # minValue <= tmp <= maxValue
                vi.append(tmp)

            self._vlist.append(vi)

    def __crossover(self):
        self._ulist = [] # 将类里的ulist置为空集
        self._CRList = []
        for i in range(self._population):
            # 为当前个体计算CR
            # !!! CR > 1 令 CR = 1  CR < 0 令 CR = 0
            CR = random.normalvariate(self._uCR, 0.1)
            if CR < 0:
                CR = 0
            if CR > 1:
                CR = 1
            self._CRList.append(CR)
            u = []
            for j in range(self._dimension):
                r = random.random()
                if r < CR:
                    u.append(self._vlist[i][j])
                else:
                    u.append(self._group[i][j])
            self._ulist.append(u)

    def __select(self):
        self._SelectedFList = []
        self._SelectedCRList = []
        newGroup = []
        min = sys.float_info.max # 临时的min 记录这次选择过程中的最小值
        for i in range(self._population):
            if self._objectFunciton(self._group[i]) < self._objectFunciton(self._ulist[i]):
                newGroup.append(self._group[i])
            else:
                newGroup.append(self._ulist[i])
                self._SelectedCRList.append(self._CRList[i])
                self._SelectedFList.append(self._FList[i])
                if not self.__bodyInList(self._archive, self._group[i]):  # 如果外部归档里没有这个元素
                    self._archive.append(self._group[i]) # 把这个元素加到外部归档里

            newValue = self._objectFunciton(newGroup[i]) # 计算一下这个个体的目标函数值
            if newValue < self._minObjectFunctionValue: # 如果比当前类里记录的最优值要好
                self._minObjectFunctionValue = newValue # 更新类里记录的最优值
                self._bestSolution = newGroup[i] # 把这个个体记录为最优解

            if newValue < min:
                min = newValue

        self._group = newGroup
        self._eachGenerationMin.append(min)

    def __mean(self, list):
        if len(list) == 0:
            return 0.0
        sum = 0
        for i in range(len(list)):
            sum += list[i]
        return 1.0 * sum / len(list)

    def __lehmerMean(self, list):
        if len(list) == 0:
            return 0.0
        pSum = 0
        sum = 0
        for i in range(len(list)):
            pSum = pSum + pow(list[i], 2)
            sum = sum + list[i]
        return 1.0 * pSum / sum

    def __updateParam(self): # 更新参数
        self._uCR = 0.5 * self._uCR + 0.5 * self.__mean(self._SelectedCRList)
        self._uF = 0.5 * self._uF + 0.5 * self.__lehmerMean(self._SelectedFList)
        while len(self._archive) > self._population: # 如果外部归档的个数超过了规定的种群数  随机的删除一些元素
            r_index = self.__random() % len(self._archive)
            self._archive.remove(self._archive[r_index])

    def Fit(self, showGenerationPiction = False, title = "JADE"): # 这个是外部调用的函数
        _output = sys.stdout
        self.__initGroup()
        for i in range(self._generation):
            self.__variation()
            self.__crossover()
            self.__select()
            self.__updateParam()
            _output.write(f'\rcomplete percent:{i:.0f}/{self._generation:.0f}/best:{float(self._minObjectFunctionValue):.25f}')
        _output.flush()    
        if showGenerationPiction == True: # 如果需要画图
            x = []
            for i in range(self._generation):
                x.append(i + 1)
            plt.plot(x, self._eachGenerationMin, 'C4')
            plt.title(title)
            plt.xlabel('generation')
            plt.ylabel('objection_function_value')
            plt.scatter(x, self._eachGenerationMin, marker='*', s=30)
            plt.show()
        return self._bestSolution, self._minObjectFunctionValue


def testObjFunction(list):
    return list[0]**2 + list[1]**3 + list[2]**2

def testFunction_10(list): # list 是自变量向量 X = [x0,x1,...,xn]
    D = len(list)
    tmp_1 = 0.0
    tmp_2 = 0.0
    for i in range(D):
        tmp_1 = tmp_1 + pow(list[i], 2.0)
        tmp_2 = tmp_2 + math.cos(2 * math.pi * list[i])
    tmp_1 = tmp_1 / D
    tmp_1 = -0.2 * pow(tmp_1, 0.5)
    tmp_1 = -20 * math.exp(tmp_1)
        
    tmp_2 = tmp_2 / D
    tmp_2 = -1.0 * math.exp(tmp_2)

    return tmp_1 + tmp_2 + 20 + math.e

def testFunction_12(list, a=10, k=1000, m=4):
    D = len(list)
    yList = []
    uSum = 0.0
    for i in range(D):
        xi = list[i]
        yi = 1.0 + 0.25 * (xi+1)
        yList.append(yi)
        ui = 0
        if xi > a:
            ui = k * pow((xi - a), m)
        elif xi < -1.0 * a:
            ui = k * pow((-1.0 * xi - a), m)
        else:
            ui = 0
        uSum = uSum + ui
    tmp = 10 * pow(math.sin(math.pi * yList[0]), 2.0)
    for i in range(D-1):
        tmp = tmp + pow( yList[i] - 1, 2) * (1 + 10 * pow(math.sin(math.pi*yList[i+1]),2) )

    tmp = (math.pi / D) * (tmp + yList[D-1] ** 2)
    return (tmp + uSum)

def testFunction_11(list):
    tmp1 = 0
    tmp2 = 1
    for i in range(len(list)):
        tmp1 =  tmp1 + ( pow ( list[i] , 2 ) ) * ( 1 / 4000)
        tmp2 = tmp2 * math.cos ( list[i] / pow( (i + 1) , 0.5 ) )
    return tmp1 - tmp2 + 1

if __name__ == "__main__":
    '''
    limit = []
    for i in range(100):
        limit.append([-32, 32])
    jade = MyJADE(testFunction_10, limit, 400, 3000)
    s, b = jade.Fit(True)
    str = "最优解:["
    for i in range(len(s)):
        str += " {}".format(s[i])
    str += "] 目标函数值:[{}]".format(b)
    print(str)
    '''
    limit = []
    for i in range(2):
        limit.append([-600 , 600])
    #[[9,80],[5,90]]
    jade = MyJADE(testFunction_11, limit, 30, 500)
    s , b = jade.Fit(True)
    str = "最优解:["
    for i in range(len(s)):
        str += " {}".format(s[i])
    str += "] 目标函数值:[{}]".format(b)
    print(str)