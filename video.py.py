import cv2
import numpy as np
import os
from cvzone.HandTrackingModule import HandDetector
import re
from keras.saving import load_model
import mediapipe as mp
import math

# ---------------------------------------

def is_line(lst):
    try:
        m=0
        x1,y1,x2,y2 = lst[0][0],lst[0][1],lst[-1][0],lst[-1][1]
        if (x2-x1)!=0:
            m = (y2-y1)/(x2-x1)
        num = np.sqrt(int(m**2+1))
        d = np.abs((lst[:,0]*m-m*x1+y1-lst[:,1])/num)
        if len(d[d>10])>5:
            return False
    except Exception as e:
        pass
    return True

def is_circle(lst):
    if np.sqrt(np.sum(np.square(lst[0]-lst[-1])))<10:
        D = np.sqrt(np.sum(np.square(lst-lst[0]),axis=1))
        r=np.max(D)/2
        max_d = lst[np.argmax(D)]
        mid_point = (lst[0][0]+max_d[0])/2,(lst[0][1]+max_d[1])/2
        d = np.sqrt(np.sum(np.square(lst-mid_point),axis=1))
        d = np.abs(d-r)
        if len(d[d<(r/4)])/len(d)>0.6:
            return True,lst[np.argmax(D)]
    return False,None

# -------------------------------------------------

active_options = [1,1]
def select_option(x1,y1):
    global img,img_output,black_img,active_options,undo_stack
    if y1<50:
        if x1<100:
            print("erase all")
            img[:] = black_img
            var_lst.clear()
            img_output[:] = black_img
        elif x1<200:
            print("erase")
            dct["thickness"] = 20
            dct["parameters"] = "erase"
            active_options[0] = 2
            active_options[1] = 1
        elif x1<280:
            print("marker")
            dct["thickness"] = 5
            dct["parameters"] = "marker"
            active_options[1] = 2
            active_options[0] = 1
        elif x1<400:
            if len(undo_stack):
                img[:] = undo_stack.pop()
            img_output[:] = black_img
            print("undo")
        
def nav_bar(nav):
    global thickness_ball,nav_img,active_options
    if nav:
        nav_img[:]=(0,0,0)
        color = (0,255,255)
        cv2.putText(nav_img, f'earse all', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,1, cv2.LINE_AA)
        cv2.putText(nav_img, f'erase', (130,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,1*active_options[0], cv2.LINE_AA)
        cv2.putText(nav_img, f'marker', (200,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,1*active_options[1], cv2.LINE_AA)
        cv2.putText(nav_img, f'undo', (280,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,1, cv2.LINE_AA)

        cv2.putText(nav_img, f'thickness :{dct['thickness']}', (380,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1, cv2.LINE_AA)
        cv2.line(nav_img,(500,40),(800,40),(255,255,255),2)
        cv2.circle(nav_img,(thickness_ball,40),10,(0,0,255),-1)

        cv2.line(nav_img,(0,150),(width,150),(255,255,255),2)

    return  (0,nav_img)

thickness_ball = 500
def thickness_bar(x1,y1):
    global thickness_ball
    if y1<60 and x1>500 and x1< 800:
        thickness_ball = x1
        thickness = np.abs(int((x1-400)/20))
        dct['thickness'] = thickness

def get_final_img_output(text,nums):
    if len(text) > 1:
        first_digit = nums[text[0]]
        for i in range(1,len(text)):
            second_digit = nums[text[i]]
            h1, h2 = first_digit.shape[0], second_digit.shape[0]
            if text[i-1] == "-":
                pad_top = (h2 - h1) // 2
                pad_bottom = h2 - h1 - pad_top
                first_digit = cv2.copyMakeBorder(first_digit, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            elif text[i] == "." :
                pad_top = (h1 - h2) // 2
                pad_bottom = h1 - h2 - pad_top
                second_digit = cv2.copyMakeBorder(second_digit, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                if h1 < h2:
                    new_width = int(second_digit.shape[1] * (h1 / h2))
                    second_digit = cv2.resize(second_digit, (new_width, h1))
                elif h2 < h1:
                    new_width = int(first_digit.shape[1] * (h2 / h1))
                    first_digit = cv2.resize(first_digit, (new_width, h2))

            padding = np.zeros((first_digit.shape[0], 10, first_digit.shape[2]), dtype=np.uint8)
            first_digit = cv2.hconcat([first_digit,padding, second_digit])
        return first_digit
    
    return nums[text]



def show_answer(output_text,nums,i,j):
    global img_output,width
    i=i+50
    j=j+10
    final_output = get_final_img_output(output_text,nums)
    limit = width-j
    if final_output.shape[1]>limit and final_output.shape[1]<limit*1.4:
        new_height = int(final_output.shape[0] * (limit / final_output.shape[1]))
        final_output = cv2.resize(final_output,(limit,new_height))
    elif final_output.shape[1]>limit and final_output.shape[1]>limit*1.4:
        i=i+150
        j=j-final_output.shape[1]
    end_i = min(i + final_output.shape[0], img_output.shape[0])
    end_j = min(j + final_output.shape[1], img_output.shape[1])
    rows = end_i - i
    cols = end_j - j
    img_output[i:end_i, j:end_j] = final_output[:rows, :cols]

def insert_multiplication(expr):
    expr = re.sub(r'(\d)(\()', r'\1*\2', expr)
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])(\()', r'\1*\2', expr)
    return expr


def clean_variables(text,indices):
    global nav_img,var_lst
    lines = text.strip().split(',')
    pattern = r"[a-zA-Z]+\s*=\s*\d+"
    unmatched = []
    var_lst = []
    required_indices = []
    for i in range(len(lines)):
        if re.fullmatch(pattern, lines[i]):
            exec(lines[i],globals())
            var_lst.append(lines[i])
        else:
            if len(indices)>i:
                required_indices.extend(indices[i:i+lines[i].count("=")])
            unmatched.append(lines[i])
    return ("".join(unmatched),required_indices)



def divide_expressions(text1):
    expressions = text1.split("=")
    for i in range(len(expressions)-1):
        expressions[i] +="="
    return expressions

def warn(e):
    cv2.putText(nav_img, "Please draw correcly", (680,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),1, cv2.LINE_AA)
    print("calculate")
    print(e)

def calc_output(text):
    output_text = eval(text[:-1])
    if output_text == int(output_text):
        output_text = str(int(output_text))
    else:
        output_text = format(output_text, '.1f')
    return output_text
    

def calculate(text1,nums,indices):
    global img_output
    nav_bar(1)
    text1 ,indices = clean_variables(text1,indices)
    full_text = ""
    expressions = divide_expressions(text1)
    img_output[:] = black_img
    for i in range(len(expressions)):
        text = insert_multiplication(expressions[i])
        full_text = full_text+text
        try:
            if len(text)>1 and text[-1] == "=" :                
                output_text = calc_output(text)
                text = text+output_text
                full_text = full_text+output_text+","
                show_answer(output_text,nums,indices[i][0],indices[i][1])
        except Exception as e:
            warn(e)

    cv2.putText(nav_img, full_text, (50,120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,255),2, cv2.LINE_AA)

def process_img(digit):
    gray_digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    pad_digit = np.pad(gray_digit, pad_width=((20, 20), (20, 20)), mode='constant', constant_values=0)
    resized_digit = cv2.resize(pad_digit,(28,28))
    fin_digit = np.where(resized_digit>50,255,0)
    return fin_digit
    
# def sort_contours(contours, row_threshold=10):
#     bounding_boxes = [cv2.boundingRect(c) for c in contours]
#     rows = []
#     for i, box in enumerate(bounding_boxes):
#         x, y, w, h = box
#         placed = False
#         for row in rows:
#             if abs(row[0][1] - y) < row_threshold:
#                 row.append((x, y, i))
#                 placed = True
#                 break
#         if not placed:
#             rows.append([(x, y, i)])
#     sorted_indices = []
#     for row in sorted(rows, key=lambda r: r[0][1]):
#         row_sorted = sorted(row, key=lambda x: x[0])
#         sorted_indices.extend([x[2] for x in row_sorted])
#     return [contours[i] for i in sorted_indices]

def sort_bowding_boxes(bounding_boxes,row_threshold=0):
    x5,y5,w5,h5 = bounding_boxes[0]
    min = np.sqrt((x5**2)*0.1+(y5**2))
    for x6,y6,_,h6 in bounding_boxes:
        if np.sqrt((x6**2)*0.1+(y6**2))<min:
            x5,y5,h5 = x5,y6,h6
    
    thresold = y5+h5-row_threshold
    # cv2.line(line_img,(0,thresold+100),(width,thresold+100),(50,50,50),1)
    lst1,lst2 = [],[]
    for i in range(len(bounding_boxes)):
        if bounding_boxes[i][1]<thresold:
            lst1.append(bounding_boxes[i])
        else:
            lst2.append(bounding_boxes[i])
    lst1 = sorted(lst1, key=lambda x: x[0])
    if len(lst2) ==0:
        return lst1
    lst1.extend(sort_bowding_boxes(lst2,row_threshold=0))
    return lst1


def detect_img(img1): 
    global digit5,nav,nums,fin_digit
    bounding_boxes,lst,prev = 0, [], " "

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    try:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        outer_contours = [c for i, c in enumerate(contours) if hierarchy[i][3] == -1]
        bounding_boxes = sort_bowding_boxes([cv2.boundingRect(c) for c in outer_contours])
    except:
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = sort_bowding_boxes([cv2.boundingRect(c) for c in contours])
    indices = []
    for x2, y2, w, h in bounding_boxes:
        digit5 = img1[y2:y2+h, x2:x2+w]
        fin_digit = process_img(digit5)
        pred = model.predict(fin_digit.reshape(1,28,28))    
        y_pred = str(np.argmax(pred))
        try:
            
            if prev == '-' and symbles.get(y_pred)=='-':
                lst.pop()
                lst.append('=')
                y_pred = "="
                indices.append((y2,x2+w))
            elif int(y_pred)>=10:
                lst.append(symbles[y_pred]) 
                y_pred = symbles[y_pred]
            else:
                lst.append(y_pred)
                if np.max(pred)==1:
                    nums[y_pred] = digit5.copy()
            prev = y_pred
        except Exception as e:
            print("detect_img")
            print(e)
    text = "".join(lst) 
    calculate(text,nums,indices)

x1_start,y1_start = 0,0
trigger,nav = 0,1
prev_x,prev_y = 0,0


# ------------------------------------------


def track(event,x,y):
    global x1_start, y1_start
    global img,img_show,img_nav_bar,img_pointer
    global height,width,background_color,count,count1
    global trigger,x_near,y_near,dct,points,tab,undo_stack
    global prev_img,nav,prev_x, prev_y ,img_hand

    if event==1:
        points.clear()
        x1_start,y1_start,trigger = x,y,1
        select_option(x1_start,y1_start)
        prev_img[:] = img
    elif event == 4:
        if dct["parameters"]=="marker" and y1_start>100:
            if len(undo_stack) > 20:  
                undo_stack.pop(0)
            undo_stack.append(img.copy())
            detect_img(img[150:,])
        trigger=0
        prev_x, prev_y = 0, 0 

    if count1%10==0:
        count1=0
        x_near,y_near=x,y  
    if trigger or tab:
        if y1_start<100:
            nav = 1
            thickness_bar(x,y)
        elif dct["parameters"]=="marker":
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            cv2.line(img, (prev_x, prev_y), (x, y), dct['color'], dct['thickness'])
            prev_x, prev_y = x, y
            points.append((x,y))
            if ((abs(x_near - x) < 5 and abs(y_near - y) < 5) or tab) and len(points) > 10:
                if count >=5 or tab:
                    count = 0 
                    if is_line(np.array(points)):
                        img[:] = prev_img
                        cv2.line(img,(x1_start,y1_start),(x,y),dct['color'],dct['thickness'])
                    else:
                        is_c = is_circle(np.array(points))
                        if is_c[0]:
                            img[:] = prev_img
                            cv2.circle(img,((is_c[1][0]+x1_start)//2,(is_c[1][1]+y1_start)//2),int((np.sqrt((is_c[1][0]-x1_start)**2+(is_c[1][1]-y1_start)**2))/2),dct['color'],dct['thickness'])
                count +=1
            count1=count1+1
        elif dct["parameters"] == "erase":
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            cv2.line(img_output, (prev_x, prev_y), (x, y), (0,0,0), dct['thickness'])
            cv2.line(img, (prev_x, prev_y), (x, y), (0,0,0), dct['thickness'])
            prev_x, prev_y = x, y
        
        tab = 0
    else:
        prev_x, prev_y = 0, 0
    cv2.putText(nav_img, ",".join(var_lst), (400,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1, cv2.LINE_AA)
    cv2.circle(img_pointer,(x,y),dct['thickness'],(0,0,255),-1) 



# ----------------------------------------------

symbles = {"10":'(',"11":')',"12":'/',"13":'*',"14":'+',
           "15":'-',"16":",","17":".","18":"y","19":"x"}
nums,var_lst ={}, []
for i in os.listdir("nums"):
    nums[i[0]] = cv2.imread(f"nums/{i}")
model = load_model("new_num.keras")

x1_start,y1_start,prev_x,prev_y = 0,0,0,0
division = 40

trigger,x_near,y_near= 0,0,0
count,count1,nav = 0,0,1
thickness_ball = 12*division
img_nav_bar = ""
points =[]
smooth_x,smooth_y = 0,0


# --------------------------------------------

height,width = 750,1050
background_color = (0,0,0)
tab = 0
dct = {"parameters" : "marker","thickness":5,"color":(255,255,255)}
img = np.full((height,width,3),background_color,dtype=np.uint8)
nav_img = np.full((150,width,3),background_color,dtype=np.uint8)
img_show = img.copy()
prev_img = img.copy()
img_pointer = img.copy()
img_output,black_img = img.copy(),img.copy()
undo_stack = [np.full((height,width,3),background_color,dtype=np.uint8)]
img_hand = np.full((height+200,width+200,3),background_color,dtype=np.uint8)
detector = HandDetector(detectionCon=0.8 , maxHands=2)
video = cv2.VideoCapture(0)
prev_lenght,min_length = 0,50
vid = cv2.VideoCapture(0)
patience,co,flag1,flag2 =0,0,False,False
handLms = 0
thumb_tip1, thumb_tip2 = 0,0
index_tip1, index_tip2 = 0,0
info = 0



mpHands = mp.solutions.hands
hands_detector = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
mpDraw = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)

def draw_hand_landmarks(img, hand_landmarks):
    mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

while True:
    key = cv2.waitKey(1)
    try:

        if key == 9:
            tab = 1
        elif key == 0:
            break

    # -------------------------------------------        
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (width+100, height+100))

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)
        img_hand = frame.copy()
        hands = []

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h_c, w_c, _ = img_hand.shape
                    cx, cy = int(lm.x * w_c), int(lm.y * h_c)
                    lmList.append((cx, cy))
                hands.append({"lmList": lmList})

        if hands:
            lmlist = hands[0]
            x_c = (lmlist["lmList"][8][0] + lmlist["lmList"][4][0]) // 2
            y_c = (lmlist["lmList"][8][1] + lmlist["lmList"][4][1]) // 2

            fingers = []
            tip_ids = [4, 8, 12, 16, 20]
            fingers.append(1 if lmlist["lmList"][4][0] > lmlist["lmList"][3][0] else 0)
            for id in range(1, 5):
                fingers.append(1 if lmlist["lmList"][tip_ids[id]][1] < lmlist["lmList"][tip_ids[id] - 2][1] else 0)

            thumb_tip1, thumb_tip2 = lmlist["lmList"][4]
            index_tip1, index_tip2 = lmlist["lmList"][8]
            length = math.hypot(index_tip1 - thumb_tip1, index_tip2 - thumb_tip2)
            info = ((thumb_tip1 + index_tip1) // 2, (thumb_tip2 + index_tip2) // 2)
            smooth_x = int(smooth_x * 0.8 + ( x_c ) * 0.2)
            smooth_y = int(smooth_y * 0.8 + ( y_c ) * 0.2)

            if (prev_lenght >= min_length and length < min_length):
                event=1
                flag1=True
            elif (prev_lenght < min_length and length >= min_length):
                event=4
                flag1=False
            else:
                event=0
            # -------------------------------------------------------------
            # print(fingers)
            if flag2:
                if fingers == [0,0,0,0,0] or fingers == [1,0,0,0,0]:
                    patience+=1
                if (patience<50 and patience>10) and fingers==[0,1,1,1,1]:
                    img[:] = black_img
                    var_lst.clear()
                    img_output[:] = black_img
                    nav_bar(1)
                    flag2=False
                if patience<50 and patience>10:
                    cv2.rectangle(img_hand, (0+10,102+10), (width-10,height-10),(255,255,255), 10)
            elif fingers == [0,1,1,1,1]:
                flag2=True
                patience=0  
            elif patience>50:
                flag2=False
            # ---------------------------------------------------------------------
            if flag1:
                cv2.putText(img_hand,'activate',(20,460), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
            else:
                cv2.putText(img_hand,'',(20,460), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
            prev_lenght = length
            track(event,smooth_x,smooth_y)

        nav,img_nav_bar = nav_bar(nav)
        img_show = cv2.addWeighted(img, 1, img_pointer, 1, 0)
        img_show = cv2.add(img_output,img)
        img_show = cv2.addWeighted(img_show, 1,img_hand[:height,:width], 0.5, 0)
        img_show[:150] = img_nav_bar
        
        if hands:
            cv2.line(img_show, (thumb_tip1, thumb_tip2), (index_tip1, index_tip2), (0, 255, 255), 3)
            cv2.circle(img_show, info, 5, (0, 0, 255), -1)
        
        img_pointer[:] = (0,0,0)
    # ----------------------------------------------
    except Exception as e:
        print('invalid value')
        print(e)
    # cv2.imshow("Frame",img_hand)
    
    cv2.imshow("drawing_pad",img_show)
cv2.destroyAllWindows()

