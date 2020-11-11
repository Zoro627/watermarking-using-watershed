import numpy as np
import cv2


picture_name = "a.jpg"
embed_m = 45      #Embedded strength
seed_rad = 4      #seed_rad is the radius of the circle
circle_cnt = 8      #Number of vertical and horizontal circles. The number of circles depicted is the square of circle_cnt.
label_cnt = circle_cnt**2 #Number of divided areas

# embed_bit= ["00", "01", "00", "00", "11", "00", "00", "00",
#             "00", "00", "00", "00", "00", "00", "00", "00",
#             "00", "00", "00", "00", "00", "00", "00", "00",
#             "00", "00", "00", "00", "00", "00", "00", "00",
#             "00", "00", "00", "00", "00", "00", "00", "00",
#             "00", "00", "00", "00", "00", "00", "00", "00",
#             "00", "00", "00", "00", "00", "00", "00", "00",
#             "00", "00", "00", "00", "00", "00", "00", "00"]
embed_bit = np.full(label_cnt, "00")
embed_bit[1]="01"
embed_bit[5]="11"
embed_bit[7]="10"
embed_bit[9]="11"
################################################################################
################################################################################
################################################################################
#PSNR calculation
def calculate_pnsr(img1, img2):
    height, width = img1.shape
    mse = 0 # (MSE)

    
    for y in range(height):
        for x in range(width):
            #mse += (img1[x, y] - float(img2[x, y])) ** 2
            mse += ( (float(img1[x, y]) - img2[x, y]) ) ** 2 / (height * width)

    
    psnr = 20 * np.log10(255/np.sqrt(mse))

    return psnr

################################################################################
################################################################################
################################################################################
#Smear transform
def sumia_transform_uto(date, alpha = 1):
    
    pi = np.pi
    e = np.exp(1)
    L = date.shape[0]
    #j = np.sqrt(-1)

    if L % 2 == 0:
        LL = int(L / 2)
    else:
        LL = int((L-1) / 2)

    theta = np.zeros(L)
    for k in range(0, LL):
        theta[k] = 2 * pi * alpha * 1/L * k**2
    for k in range(LL, L):
        theta[k] = -2 * pi * alpha * 1/L * (L-k)**2

    s = np.zeros((L,L), dtype=np.complex128)
    for k in range(0, L):
        s[k, k] = np.exp(1j * theta[k])

    date = np.fft.ifft( np.dot(np.fft.fft(date), s) )
    # date = np.fft.fft(date)
    # date = np.dot(s, date.T)
    # date = np.fft.ifft(date)

    return date

################################################################################
################################################################################
################################################################################
#Desmear transform
def desumia_transform_uto(date, alpha = 1):
    pi = np.pi
    e = np.exp(1)
    L = date.shape[0]
    #j = np.sqrt(-1)

    if L % 2 == 0:
        LL = int(L / 2)
    else:
        LL = int((L-1) / 2)

    theta = np.zeros(L)
    for k in range(0, LL):
        theta[k] = 2 * pi * alpha * 1/L * k**2
    for k in range(LL, L):
        theta[k] = -2 * pi * alpha * 1/L * (L-k)**2

    s = np.zeros((L,L), dtype=np.complex128)
    for k in range(0, L):
        s[k, k] = np.exp(-1j * theta[k])

    date = np.fft.ifft( np.dot(np.fft.fft(date), s) )
    # date = np.fft.fft(date)
    # date = np.dot(s, date.T)
    # date = np.fft.ifft(date)

    return date

################################################################################
################################################################################
################################################################################
#A function that stores the area of label_number in a one-dimensional array in raster scanning.
def raster_in(img, markers, label_number):
        height, width = img.shape
        date = np.empty(0)

        for y in range(height):
            for x in range(width):
                if markers[x, y] == label_number:
                    date = np.append(date, img[x, y])

        return date

################################################################################
################################################################################
################################################################################

#Calculate the value to embed and the location
def embed_location_value(date, label_number):
    max = np.max(date)
    embed_max = 0
    embed_location = 0
    embed_value = 0
    L = date.shape[0]
    embed_strength = embed_m

    if embed_bit[label_number-1] == "00":
        for i in range(0, L, 4):
            if embed_max < date[i]:
                embed_max = date[i]
                embed_location = i
        embed_value = max - embed_max + embed_strength
    elif embed_bit[label_number-1] == "01":
        for i in range(1, L, 4):
            if embed_max < date[i]:
                embed_max = date[i]
                embed_location = i
        embed_value = max - embed_max + embed_strength
    elif embed_bit[label_number-1] == "10":
        for i in range(2, L, 4):
            if embed_max < date[i]:
                embed_max = date[i]
                embed_location = i
        embed_value = max - embed_max + embed_strength
    elif embed_bit[label_number-1] == "11":
        for i in range(3, L, 4):
            if embed_max < date[i]:
                embed_max = date[i]
                embed_location = i
        embed_value = max - embed_max + embed_strength


    return embed_location, embed_value


################################################################################
################################################################################
################################################################################

#Function to embed a watermark
def watermark_embed(img, markers):
    height, width = img.shape

    for label_number in range(1, label_cnt+1):
        #Store the pixels of the area number label_number in a one-dimensional array
        date = raster_in(img, markers, label_number)

        embed_date = np.zeros(date.shape[0])

        #sumia_transform
        date = sumia_transform_uto(date.copy())

        #Calculate the value to embed and the location
        embed_location, embed_value = embed_location_value(date, label_number)
        embed_date[embed_location] =  embed_date[embed_location] + embed_value

        date = date + embed_date

        #desumia_transform
        date = desumia_transform_uto(date.copy())
        date = np.real(date)

        #Return the pixel to its original position
        for y in range(height):
            for x in range(width):
                if markers[x, y] == label_number:
                    img[x, y] = date[0]
                    date = np.delete(date, 0)

    return img

################################################################################
################################################################################
################################################################################

def watermark_detection(img, markers):
    error_cnt = 0

    for label_number in range(1, label_cnt+1):
        date = raster_in(img, markers, label_number)

        #sumia_transform
        date = sumia_transform_uto(date)

        #Calculate watermark with mod4
        if np.argmax(date) % 4 == 0:
            print("00", "...label_number == ", label_number)
            if embed_bit[label_number-1] != "00":
                error_cnt += 1
        elif np.argmax(date) % 4 == 1:
            print("01", "...label_number == ", label_number)
            if embed_bit[label_number-1] != "01":
                error_cnt += 1
        elif np.argmax(date) % 4 == 2:
            print("10", "...label_number == ", label_number)
            if embed_bit[label_number-1] != "10":
                error_cnt += 1
        elif np.argmax(date) % 4 == 3:
            print("11", "...label_number == ", label_number)
            if embed_bit[label_number-1] != "11":
                error_cnt += 1

    print(" ", error_cnt," ")
    print(" ", (label_cnt - error_cnt) / label_cnt * 100, "%")




################################################################################
################################################################################
################################################################################
################################################################################
img = cv2.imread(picture_name, cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_border = img_gray.copy()
h, w, _ = img.shape
dsp_img = img_gray.copy()
markers = np.zeros((h,w,1), np.int32)

#Draw a circle on the marker image and confirmation image
seed_num = 0
x = int(h/(circle_cnt*2))
y = int(w/(circle_cnt*2))
for none1 in range(circle_cnt):
    for none2 in range(circle_cnt):
        seed_num += 1
        cv2.circle(markers, (x, y), seed_rad, seed_num, -1, 8, 0)
        cv2.circle(dsp_img, (x, y), seed_rad, 255, -1, 8, 0)
        x += int(h/circle_cnt)
    y += int(w/circle_cnt)
    x = int(h/(circle_cnt*2))

#Run Watershed Segmentation
markers = cv2.watershed(img, markers)

#Color the border to display (for display)
for height in range(h):
     for width in range(w):
         if markers[height, width] == -1:
             img_border[height, width] = 255   

#Watermark embedding
embed_img = watermark_embed(img_gray.copy(), markers)

#Watermark detection
watermark_detection(embed_img.copy(), markers)

#Writing image
cv2.imwrite("embed.bmp", embed_img)
cv2.imwrite("embed_border.bmp", img_border)
cv2.imwrite("src_grey.bmp", img_gray)

#Calculation and display of PNSR
print("PNSR",calculate_pnsr(img_gray, embed_img), "dB")

#Image display
cv2.imshow('input', img_gray)
cv2.imshow('output', embed_img)
cv2.imshow('border', img_border)
cv2.imshow('circle_image', dsp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
