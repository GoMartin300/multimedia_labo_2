import sys
import cv2
import numpy as np


const_amount_of_pictures = 6
const_max_pixel_value = 255
const_min_pixel_value = 0
const_delta = [14]
maxDelta = 20
dynamic_on = False
single_test = False


def calc_rmse(orgimg, comimg):
    """
    calculates the RMSE for 2 images
    :param comimg: reconstructed image
    :param orgimg: original image
    :return: the calculated RMSE value
    """
    # calculate RMSE
    height = len(orgimg)
    width = len(orgimg[0])
    tempvalues = np.subtract(orgimg, comimg)
    tempvalues = np.power(tempvalues.astype('uint'), 2)
    return np.power(np.divide(np.sum(tempvalues), height * width), 0.5)


def delta_compress(orgimg, delta, adaptive):
    """
    compresses an image using delta compression
    :param orgimg: the original image
    :return: a 2d list containing the 0 and 1 values needed to recreate the image
    """

    lastBit = 0  # what is the sign of the previous pixel? -delta or + delta?
    repeatAmmount = 0  # how many times has it been the same sign bit?
    adaptiveAmm = 1  # the factor we multiply with

    altern = 0
    height = orgimg.shape[0]
    width = orgimg.shape[1]
    temp = []
    tempcomimg = np.zeros((height, width), dtype='int')
    tempcomimg[0, 0] = orgimg[0, 0]
    comtable = np.zeros((height, width), dtype='int')
    for h in range(0, height):

        check_above = (h > 0)
        if check_above:
            if tempcomimg[h - 1, 0] < orgimg[h, 0]:
                comtable[h, 0] = 1
            elif tempcomimg[h - 1, 0] == orgimg[h, 0]:
                comtable[h, 0] = altern
                altern = (altern + 1) % 2
            if comtable[h, 0] == 1:
                tempcomimg[h, 0] = tempcomimg[h - 1, 0] + delta
            else:
                tempcomimg[h, 0] = tempcomimg[h - 1, 0] - delta

        for w in range(1, width):
            temp.clear()
            if check_above:
                temp.append(tempcomimg[h - 1, w])
            temp.append(tempcomimg[h, w - 1])
            average = np.sum(temp)//len(temp)

            if average < orgimg[h, w]:
                comtable[h, w] = 1
            elif average == orgimg[h, w]:
                comtable[h, w] = altern
                altern = (altern + 1) % 2
            if comtable[h, w] > 0:
                lastBit, repeatAmmount, adaptiveAmm = delta_adaptive(lastBit,repeatAmmount,1,adaptive)
                tempcomimg[h, w] = tempcomimg[h, w - 1] + delta*adaptiveAmm
                if tempcomimg[h, w] > const_max_pixel_value:
                    tempcomimg[h, w] = const_max_pixel_value
            else:
                lastBit, repeatAmmount, adaptiveAmm = delta_adaptive(lastBit, repeatAmmount, 0, adaptive)
                tempcomimg[h, w] = tempcomimg[h, w - 1] - delta*adaptiveAmm
                if tempcomimg[h, w] < const_min_pixel_value:
                    tempcomimg[h, w] = const_min_pixel_value
    return comtable


def delta_adaptive(lastBit, repeatAmmount,currentBit, adaptive):
    adaptiveAmm = 1
    if adaptive:
        if currentBit == lastBit:
            repeatAmmount = repeatAmmount + 1
            adaptiveAmm = repeatAmmount/8 + 1  # int makes sure it's rounded
        else:
            lastBit = (lastBit + 1) % 2
            repeatAmmount = 0
    return lastBit, repeatAmmount, adaptiveAmm


def delta_decompress(comtable, delta, adaptive):
    """
    recreates an image based on the value of the original image pixel on location 0,0 and the acompanying comptable and
    a given delta value
    :param comtable: table containing the 0 and 1 values
    :param delta:
    :return: returns the reconstructed image
    """

    lastBit = 0  # what is the sign of the previous pixel? -delta or + delta?
    repeatAmmount = 0  # how many times has it been the same sign bit?
    adaptiveAmm = 1  # the factor we multiply with

    height = comtable.shape[0]
    width = comtable.shape[1]
    tempcomimg = np.zeros((height, width), dtype='int')
    check_above2 = False
    for h in range(0, height):
        if check_above2:
            if comtable[h, 0] == 1:
                tempcomimg[h, 0] = comtable[h - 1, 0] + delta
            else:
                tempcomimg[h, 0] = comtable[h - 1, 0] - delta
        else:
            check_above2 = True
            tempcomimg[0, 0] = 127
        for w in range(1, width):
            if comtable[h, w] > 0:
                lastBit, repeatAmmount, adaptiveAmm = delta_adaptive(lastBit, repeatAmmount, 1, adaptive)
                tempcomimg[h, w] = tempcomimg[h, w - 1] + delta*adaptiveAmm
                if tempcomimg[h, w] > const_max_pixel_value:
                    tempcomimg[h, w] = const_max_pixel_value
            else:
                lastBit, repeatAmmount, adaptiveAmm = delta_adaptive(lastBit, repeatAmmount, 0, adaptive)
                tempcomimg[h, w] = tempcomimg[h, w - 1] - delta*adaptiveAmm
                if tempcomimg[h, w] < const_min_pixel_value:
                    tempcomimg[h, w] = const_min_pixel_value
    comimg = tempcomimg.astype('uint8')
    return comimg


def calculations(orgimg_list, delta):
    """
    calculations combines all other functions so that for each delta the compressed images and RMSE values can be easily
    retrieved
    :param orgimg_list: list containing the original pictures
    :return: returns the list with the rmse values for all pictures and a list with the corresponding reconstructed
    images
    """
    comtables = []
    comimg_list = []
    rmse_list = []
    for i in range(0, const_amount_of_pictures):
        comtables.append(delta_compress(orgimg_list[i], delta[i], dynamic_on))
        comimg_list.append(delta_decompress(comtables[i], delta[i], dynamic_on))
        rmse_list.append(calc_rmse(orgimg_list[i], comimg_list[i]))
        print('completed ' + str(i + 1) + '/' + str(const_amount_of_pictures))
    return rmse_list, comimg_list


def delta_calculate_meth1(orgimg_list):
    lowestRMSE = []
    used_delta = []
    for j in range(0, const_amount_of_pictures):
        lowestRMSE.append(100)
        used_delta.append(0)
        for i in [x * 2 for x in range(1, maxDelta//2)]:
            compr = delta_compress(orgimg_list[j], i, dynamic_on)
            compr = delta_decompress(compr, i, dynamic_on)
            newRMSE = calc_rmse(orgimg_list[j], compr)
            print("rmse pic" + str(j) + "on delta:" + str(i) + " result rmse is:"+str(newRMSE) + "\n")
            if newRMSE < lowestRMSE[j]:
                lowestRMSE[j] = newRMSE
                used_delta[j] = i

            # is there a patern? if so ...
        print("FINAL DELTA IS "+str(used_delta)+"\n")
    return used_delta


orgimg_list = []
for i in range(1, const_amount_of_pictures + 1):
    orgimg_list.append(cv2.imread(str(sys.argv[i]), 0))

for i in range(0, len(orgimg_list)):
    cv2.imshow('original image ' + str(i), orgimg_list[i])
    cv2.waitKey(delay=1)
if single_test:
    rmse_list, comimg_list = calculations(orgimg_list, const_delta)
else:
    rmse_list, comimg_list = calculations(orgimg_list, delta_calculate_meth1(orgimg_list))


for i in range(0, len(comimg_list)):
    cv2.imshow('result' + str(i), comimg_list[i])
    cv2.waitKey(delay=1)
    print(str(rmse_list[i]) + "\n")

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cv2.destroyAllWindows()
