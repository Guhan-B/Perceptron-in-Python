import cv2 as cv
import numpy as np
import random
import keyboard

def point_map(value, minA, maxA, minB, maxB):
    return int((1 - ((value - minA) / (maxA - minA))) * minB + ((value - minA) / (maxA - minA)) * maxB)

def f(x):
    # y = mx + c
    return -0.3 * x - 0.2


#Activation Function
def sign(n):
    if(n>=0):
        return 1
    else:
        return -1

class Point:
    def __init__(self,manual=False,x=0,y=0):
        if manual:
            self.x = x
            self.y = y
        else:
            self.x = random.uniform(-1,1)
            self.y = random.uniform(-1,1)

        self.bias = 1
        self.px = point_map(self.x,-1,1,0,width)
        self.py = point_map(self.y,-1,1,height,0)
        if self.y > f(self.x) :
            self.label = 1
        else:
            self.label = -1

    def getPoint(self):
        return [self.x,self.y,self.bias]

class Perceptron:
    def __init__(self):
        self.lr = 0.1
        self.weights = []
        for i in range(3):
            self.weights.append(random.uniform(-1,1))

    def guess(self,point):
        sum = 0.0
        input = point.getPoint()
        for i in range(len(self.weights)):
            sum += input[i]*self.weights[i]
        output = sign(sum)
        return output

    def train(self,point,target):
        guess = self.guess(point)
        error = target - guess
        input = point.getPoint()
        for i in range(len(self.weights)):
            self.weights[i] += error * input[i] * self.lr

    def guessY(self,x):
        m = self.weights[0] / self.weights[1]
        b = self.weights[2]
        return int(m * x + b)

brain = Perceptron()
points = []
width = 800
height = 800
point_count = 100
plt = np.zeros((width,height,3),np.uint8)

plt[:,:] = (255,255,255)

p1 = Point(True,-1,f(-1))
p2 = Point(True,1,f(1))

plt = cv.line(plt,(p1.px,p1.py),(p2.px,p2.py),(0,0,0),1)

# p3 = Point(True,-1,brain.guessY(-1))
# p4 = Point(True,1,brain.guessY(1))
#
# plt = cv.line(plt,(p3.px,p3.py),(p4.px,p4.py),(0,0,0),1)



for i in range(point_count):
    #setting points and ploting it
    points.append(Point())
    if points[i].label == 1:
        color = (100,100,100)
        thickness = 2
    else:
        color = (0,0,0)
        thickness = -1
    plt = cv.circle(plt,(points[i].px,points[i].py),10,color,thickness)

    # peceptron guessing the position
    guess = brain.guess(points[i])
    if guess == points[i].label:
        color = (0,255,0)
    else:
        color = (0,0,255)
    plt = cv.circle(plt,(points[i].px,points[i].py),5,color,thickness=-1)

cv.imshow("graph",plt)

autoTrain = True
print('press a key to start training')
cv.waitKey(0)

while not (keyboard.is_pressed('q')):
    p3 = Point(True,-1,brain.guessY(-1))
    p4 = Point(True,1,brain.guessY(1))

    plt = cv.line(plt,(p3.px,p3.py),(p4.px,p4.py),(0,0,0),1)
    if keyboard.is_pressed('t') or autoTrain:
        for point in points:
            brain.train(point,point.label)
            guess = brain.guess(point)

            if guess == point.label:
                color = (0,255,0)
            else:
                color = (0,0,255)
            plt = cv.circle(plt,(point.px,point.py),5,color,thickness=-1)

            cv.imshow("graph",plt)
            cv.waitKey(1)

cv.destroyAllWindows()
