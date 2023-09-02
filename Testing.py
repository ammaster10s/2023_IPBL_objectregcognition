import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import math
import numpy as np
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk
import Canny as canny

def distance(a,b):
    d = math.sqrt(math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2)+math.pow(a[2]-b[2],2))
    return d



"""method for checking if 2 segments intersect"""
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)



""""method for checking distance for identify good/bad posture"""
def get_hand_data(i, frame):

    tf10 = Hand.get_landmark(id_hand= i, id_landmark= 10)
    tf11 = Hand.get_landmark(id_hand= i, id_landmark= 11)
    tf15 = Hand.get_landmark(id_hand= i, id_landmark= 15)
    tf14 = Hand.get_landmark(id_hand= i, id_landmark= 14)
    ref = (distance(tf10,tf11) + distance(tf15,tf14))*0.5
    #this is reference = average of constant distance


    f8 = Hand.get_landmark(id_hand= i, id_landmark= 8)
    f12 = Hand.get_landmark(id_hand= i, id_landmark= 12)
    f16 = Hand.get_landmark(id_hand= i, id_landmark= 16)
    d1 = distance(f8,f16)/ref #distance of d1 divide by ref
    d2 = distance(f16,f12)/ref # distance of d2 divide by ref

    if(d1 > 1.0 and d2 > 0.7): # the ratio come from testing 
        return  "This is good posture ++"
    else:
        return "This is bad posture --"


# def motionDetection(i, frame, old_data, message):

#     tf10 = Hand.get_landmark(id_hand= i, id_landmark= 10)
#     tf11 = Hand.get_landmark(id_hand= i, id_landmark= 11)
#     tf15 = Hand.get_landmark(id_hand= i, id_landmark= 15)
#     tf14 = Hand.get_landmark(id_hand= i, id_landmark= 14)
#     ref = (distance(tf10,tf11) + distance(tf15,tf14))*0.5

#     dist = comparison(i, old_data, 13) + comparison(i, old_data, 14) + comparison(i, old_data, 15) + comparison(i, old_data, 17) + comparison(i, old_data, 18) + comparison(i, old_data, 19) / 6
#     if int(dist / ref * 2) > 1:
#         return "This is bad posture --"
#     else:
#         return message

def comparison(i, old_data, landmark):
    dist = 0
    index = 0
    for k, data in enumerate([13, 14, 15, 17, 18, 19]):

        if data == landmark:
            index = k
    for j in range(3):
        dist = dist + abs(Hand.get_landmark(id_hand= i, id_landmark= landmark)[j] - old_data[index][j])
    return dist

def old_data_Hand(i, landmarks):
    data = []
    for landmark in landmarks:
        data.append(Hand.get_landmark(id_hand= i, id_landmark= landmark))
    return data

def bilateral(img):
    img_res = cv2.bilateralFilter(src=img, d=9, sigmaColor=75, sigmaSpace=75)
    return img_res 


"""in use"""
def hough_show(img, lines, frame):
    h, w = img.shape[:2]
    if h > w:
        longer = h * 1.4
    else:
        longer = w * 1.4

    if lines is not None:
        for tmp in lines:
            rho,theta = tmp[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + longer*(-b))
            y1 = int(y0 + longer*(a))
            x2 = int(x0 - longer*(-b))
            y2 = int(y0 - longer*(a))
            # cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
    return frame 



"""in use"""
def hough_list(img, lines):
    lineList = []
    h, w = img.shape[:2]
    if h > w:
        longer = h * 1.4
    else:
        longer = w * 1.4

    if lines is not None:
        for tmp in lines:
            rho,theta = tmp[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + longer*(-b))
            y1 = int(y0 + longer*(a))
            x2 = int(x0 - longer*(-b))
            y2 = int(y0 - longer*(a))
            lineList.append((x1, y1, x2, y2))

    return lineList

def hand_point_Maximum(i, h, w):
    xmax = 0
    xmin = w
    ymax = 0
    ymin = h

    for j in range(21):
        x = Hand.get_landmark(id_hand= i, id_landmark= j)[0]
        if x > xmax and x < w:
            xmax = x
        elif x < xmin and x > 0:
            xmin = x

        y = Hand.get_landmark(id_hand= i, id_landmark= j)[1]
        if y > ymax and y < h:
            ymax = y
        elif y < ymin and y > 0:
            ymin = y

    return xmax, xmin, ymax, ymin

# 線分ABと線分CDの交点を求める関数 #maybe not use anymore?
# def calc_cross_point(pointA, pointB, pointC, pointD):
#     cross_point = (0,0)
#     bunbo = (pointB[0] - pointA[0]) * (pointD[1] - pointC[1]) - (pointB[1] - pointA[1]) * (pointD[0] - pointC[0])

#     # 直線が平行な場合
#     if (bunbo == 0):
#         return False, cross_point

#     vectorAC = ((pointC[0] - pointA[0]), (pointC[1] - pointA[1]))
#     r = ((pointD[1] - pointC[1]) * vectorAC[0] - (pointD[0] - pointC[0]) * vectorAC[1]) / bunbo

#     # rを使った計算の場合
#     distance = ((pointB[0] - pointA[0]) * r, (pointB[1] - pointA[1]) * r)
#     cross_point = (int(pointA[0] + distance[0]), int(pointA[1] + distance[1]))

#     return True, cross_point

# 交点を描画する関数 # maybe not use anymore?
# def draw_cross_points(image, lineList):
#     size = len(lineList)
#     point = []

#     cnt = 0
#     for i in range(size-1):
#         for j in range(i+1, size):
#             pointA = (lineList[i][0], lineList[i][1])
#             pointB = (lineList[i][2], lineList[i][3])
#             pointC = (lineList[j][0], lineList[j][1])
#             pointD = (lineList[j][2], lineList[j][3])
#             ret, cross_point = calc_cross_point(pointA, pointB, pointC, pointD) # 交点を計算
#             if ret:
#                 # 交点が取得できた場合でも画像の範囲外のものは除外
#                 if (cross_point[0] >= 0) and (cross_point[0] <= image.shape[1]) and (cross_point[1] >= 0) and (cross_point[1] <= image.shape[0]) :
#                     cv2.circle(image, (cross_point[0],cross_point[1]), 2, (255,0,0), 3) # 交点を青色で描画
#                     cnt = cnt + 1
#                     point.append(cross_point)
#     return point 

# def triangle(point, message):
#     f5 = Hand.get_landmark(id_hand= i, id_landmark= 5)
#     f6 = Hand.get_landmark(id_hand= i, id_landmark= 6)
#     f8 = Hand.get_landmark(id_hand= i, id_landmark= 8)
#     if len(point) >= 1:
#         if len(point) >= 2:
#             point = np.average(point, axis=1)
#             if distance(f5, f6) * 3 > distance((f8[0], f8[1], 0), (point[0], point[1], 0)) or distance(f5, f6) * 5 < distance((f8[0], f8[1], 0), (point[0], point[1], 0)):
#                 return  "This is bad posture --"
#             else:
#                 return message 
#         if distance(f5, f6) * 3 > distance((f8[0], f8[1], 0), (point[0][0], point[0][1], 0)) or distance(f5, f6) * 5 < distance((f8[0], f8[1], 0), (point[0][0], point[0][1], 0)):
#             return  "This is bad posture --"
#         else:
#             return message 
#     else:
#         return message

"""return 1 if U chopstick, 2 if L chopstick, -1 for not pass"""
def check_line_is_on_chopstick(i,line):
    # reference point for checking intersection
    refU1 = np.array(Hand.get_landmark(id_hand=i,id_landmark=8))[:2]#reference p1 for Upper chopstick
    refU2 = np.array(Hand.get_landmark(id_hand=i,id_landmark=12))[:2]
    refL1 = refU2
    refL2 = np.array(Hand.get_landmark(id_hand=i,id_landmark=16))[:2]

    if intersect(refU1,refU2,(line[0],line[1]),(line[2],line[3])):
        return 1
    elif intersect(refL1,refL2,(line[0],line[1]),(line[2],line[3])):
        return 2
    else:
        return -1

"""draw virtual chopstick based on hand data, receive 3 args ( id_hand, length of chopstick, frame to be displayed on) 
and return ref 2D vector for upper and lower chopstick """
def draw_chopstick(i,frame): 
    
    #referecne distance for drawing chopstick
    ref_d = np.linalg.norm(Hand.get_landmark(id_hand=i,id_landmark=14)-Hand.get_landmark(id_hand=i,id_landmark=13))


    #Upper Chopstick code
    f11p = np.array(Hand.get_landmark(id_hand= i, id_landmark= 11))
    f10p = np.array(Hand.get_landmark(id_hand= i, id_landmark= 10))
    vfp2 = f11p - f10p
    v_U_chopstick = vfp2/np.linalg.norm(vfp2)
    U_chopstick_mid = (np.array(Hand.get_landmark(id_hand= i, id_landmark= 8)) + np.array(Hand.get_landmark(id_hand= i, id_landmark= 12)))*0.5
    U_chopstick_p2 = np.array(U_chopstick_mid + ref_d*2.5*v_U_chopstick,np.int32)
    U_chopstick_p1 = np.array(U_chopstick_mid - ref_d*2.5*v_U_chopstick,np.int32)
    cv2.line(frame,U_chopstick_p1[:2],U_chopstick_p2[:2],[160,231,248],thickness = 5)#draw upper chopstick


    #Lower Chopstick code
    p1_L = np.array((Hand.get_landmark(id_hand=i,id_landmark=19)*2 + Hand.get_landmark(id_hand=i,id_landmark=20))/3)
    p2_L = np.array((Hand.get_landmark(id_hand=i,id_landmark=15)*2 + Hand.get_landmark(id_hand=i,id_landmark=16))/3)
    v_p2 = p2_L - p1_L
    v_p2_u = v_p2/np.linalg.norm(v_p2)
    L_chopstick_mid = np.array(p1_L + v_p2_u*ref_d*0.35,np.int32)
    # cv2.circle(frame,L_chopstick_mid[:2],5,(1,1,15),thickness = -1)

    rat = 4.5 # for fine tune lower chopstick location
    p4_L = np.array((Hand.get_landmark(id_hand=i,id_landmark=5)*(rat-1) + Hand.get_landmark(id_hand=i,id_landmark=0))/rat)
    v_L2 = p4_L - np.array(Hand.get_landmark(id_hand=i,id_landmark=17))
    v_L2_u = v_L2/np.linalg.norm(v_L2)
    L_chopstick_end = np.array(p4_L + v_L2_u*ref_d*0.3,np.int32)
    # cv2.circle(frame,L_chopstick_end[:2],5,(1,1,15),thickness = -1)
    v_L_chopstick = L_chopstick_mid - L_chopstick_end
    cv2.line(frame,np.array(L_chopstick_end - 0.5*v_L_chopstick,np.int32)[:2],np.array(L_chopstick_end + 2.5*v_L_chopstick,np.int32)[:2],[160,231,248],thickness = 5)#draw lower chopstick

    return v_U_chopstick[:2], v_L_chopstick[:2 ]

"""check if angle between 2 vectors exceed a given threshold(threshold_angle)"""
def check_vector_similiarity(v1,v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    angle  = math.acos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    if angle < threshold_angle:
        return True
    else:
        return False

correct = [False,False]

"""Input : lines_list(hough_list), ref vector for top chopstick, ref vector for bottom chopstick 
    check if a line in line list is similiar to a good posture"""
def check_chopstick_position(i,hough_list,v_upper,v_lower):
    if len(hough_list) > 0:
        for t in hough_list:
            if check_line_is_on_chopstick(i,t) == 1:
                v = np.array([t[0]-t[2],t[1]-t[3]]) # convert info from hough_list to vector
                if check_vector_similiarity(v,v_upper): # check if detected vector similiar to virtual chopstick upper
                    print("upper correct")
                    cv2.line(res,(t[0],t[1]),(t[2],t[3]),(0,255,0),2)
                    correct[0] = True
                else:
                    cv2.line(res,(t[0],t[1]),(t[2],t[3]),(0,0,255),2)
                    print("upper incorrect")
                    correct[0] = False
            if check_line_is_on_chopstick(i,t) == 2: # check if detected vector similiar to virtual chopstick lower
                v = np.array([t[0]-t[2],t[1]-t[3]])
                if check_vector_similiarity(v,v_lower):
                    print("lower correct")
                    cv2.line(res,(t[0],t[1]),(t[2],t[3]),(0,255,0),2)
                    correct[1] = True 
                else:
                    cv2.line(res,(t[0],t[1]),(t[2],t[3]),(0,0,255),2)
                    print("lower incorrect")
                    correct[1] = False

# def line(i, lines):
#     f8 = Hand.get_landmark(id_hand= i, id_landmark= 8)
#     f12 = Hand.get_landmark(id_hand= i, id_landmark= 12)
#     f16 = Hand.get_landmark(id_hand= i, id_landmark= 16)
#     if lines is not None:
#         for tmp in lines:
#             rho,theta = tmp[0] 
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a*rho
#             y0 = b*rho

cap = cv2.VideoCapture(0)
Hand = HandLmk(num_hands = 1)
dummy = 0
threshold_angle = (2*math.pi/360)*20 # if angle different more than this threshold = > not correct
while cap.isOpened():
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    flipped_frame = cv2.flip(frame.copy(), 1)#fliped frame
    Hand.detect(flipped_frame)#dectec hand in flipped frame
    annotated_frame = Hand.visualize(flipped_frame)
    res = cv2.flip(frame.copy(), 1)

    for i in range(Hand.num_detected_hands): #if detected multiple hands choose 1
        if Hand.get_handedness( id_hand = i ) == 'Right': #if the hand is right hand

            message = get_hand_data(i, flipped_frame)
            old_data = old_data_Hand(i, [13, 14, 15, 17, 18, 19])  
            bil = bilateral(res.copy()) #bilateral filter
            edges = canny.Canny(bil, dummy) #Canny edge detector
            lines = cv2.HoughLines(np.uint8(edges.copy()),1 ,np.pi/180, 150) #Hough transform
            res = hough_show(edges.copy(), lines, res.copy())

            houghList = hough_list(edges.copy(), lines) #hough list = [(x1,y1,x2,y2), ....]
            
            v_upper, v_lower = draw_chopstick(i,res)
            check_chopstick_position(i,houghList,v_upper,v_lower)
            if (correct[0] and correct[1]):
                print("yay")
            else:
                message = "This is bad posture --"
            xmax, xmin, ymax, ymin = hand_point_Maximum(i, h, w)
            # cv2.rectangle(res, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
            cv2.putText(flipped_frame,text= message ,org=(100, 300),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,color=(0, 255, 0),thickness=2,lineType=cv2.LINE_4)

    cv2.imshow('result', res)   
    # cv2.imshow('annotated frame', annotated_frame)
    cv2.imshow('flipped_frame', flipped_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
Hand.release()