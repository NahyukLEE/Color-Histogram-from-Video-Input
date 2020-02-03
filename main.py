import os, PIL
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

vidcap = cv2.VideoCapture('SampleVideo.mp4')

count = 0

while(vidcap.isOpened()): 
    ret, image = vidcap.read()
    
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(length)

    if (count < length) :
        if(int(vidcap.get(1)) % 50 == 0):
            print('Saved frame number : ' + str(int(vidcap.get(1))))
            cv2.imwrite("frame%d.jpg" % count, image)
            print('Saved frame%d.jpg' % count)
            count += 1
        else :
            count += 1
    else :
        if(int(vidcap.get(1)) % 50 == 0):
            print('Saved frame number : ' + str(int(vidcap.get(1))))
            cv2.imwrite("frame%d.jpg" % count, image)
            print('Saved frame%d.jpg' % count)
        break

vidcap.release()


allfiles=os.listdir(os.getcwd())
imlist=[filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]

w,h=Image.open(imlist[0]).size
N=len(imlist)

arr=np.zeros((h,w,3),np.float)

for im in imlist:
    imarr=np.array(Image.open(im),dtype=np.float)
    arr=arr+imarr/N

arr=np.array(np.round(arr),dtype=np.uint8)

image = Image.fromarray(arr,mode="RGB")
image.save("average.jpg")



image = cv2.imread('average.jpg', cv2.IMREAD_COLOR)
print(image.shape)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = image.reshape((image.shape[0] * image.shape[1], 3)) # height, width
print(image.shape)

# (25024, 3)

k = 5
clt = KMeans(n_clusters = k)
clt.fit(image)

for center in clt.cluster_centers_:
    print(center)

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


hist = centroid_histogram(clt)
print(hist)

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
        print(percent, color)
    # return the bar chart
    return bar

bar = plot_colors(hist, clt.cluster_centers_)


# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()

