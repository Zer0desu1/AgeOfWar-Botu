import mss
from PIL import Image
import cv2 as cv
import numpy as np
import pyautogui as pyg
import pytesseract
import time 
pytesseract.pytesseract.tesseract_cmd = r'D:/tesseract/tesseract.exe'
flag5=True
flag1=True
flag2=True
flag3=True
flag4=True
flag5=True
flag6=True
flag7=True
flag8=True
flagextra=True
flagextra2=True
evoFlag1=True
evoFlag2=False
evoFlag3=False
evoFlag4=False
spot1=True
spot2=True
spot3=True
gecis=False
extraf=True
space=13
enemy_img=cv.imread('enemy.png')
w, h = enemy_img.shape[1], enemy_img.shape[0]
enemy2_img=cv.imread('enemy2.png')
w2, h2 = enemy2_img.shape[1], enemy2_img.shape[0]


def buildTurret(turNum,locNum):
    pyg.click(1123,163)
    time.sleep(0.1)
    pyg.click(965,163)
    time.sleep(0.1)
    pyg.click(905+((turNum-1)*60),163)
    time.sleep(0.1)
    pyg.click(395,559-((locNum-1)*60))#559 first location
    time.sleep(0.1)
    pyg.click(1123,163)
    time.sleep(0.1)
    pyg.click(905,163)



def levelUp():
    pyg.click(1123,163)
    time.sleep(0.1)
    pyg.click(1123,163)
    time.sleep(0.1)
    pyg.click(905,163)
    
def buildTower():
    pyg.click(1123,163)
    time.sleep(0.1)
    pyg.click(1085,163)   
    time.sleep(0.1)
    pyg.click(905,163)
def sellTurret(locNum):
    pyg.click(1123,163)
    time.sleep(0.1)
    pyg.click(1025,163)   
    time.sleep(0.1)
    pyg.click(395,559-((locNum-1)*58))
    time.sleep(0.1)
    pyg.click(905,163)

def extract_text_from_image(image_path):
    # Read the image using OpenCV
    image = cv.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Use pytesseract to extract text
    extracted_text = pytesseract.image_to_string(gray_image)

    return extracted_text

dimensions_money = {
        'left': 363,
        'top': 131,
        'width': 90,
        'height': 26
    }
dimensions_exp = {
        'left': 390,
        'top': 160,
        'width': 88,
        'height': 26
    }
while True:
    with mss.mss() as sct:
        screenshot = sct.shot(output="screenshot.png", mon=2)  
        money = sct.grab(dimensions_money)
        exp=sct.grab(dimensions_exp)
        img = Image.open(screenshot)
        money_np = np.array(money)
        exp_np=np.array(exp)

        img_np = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
        result = cv.matchTemplate(img_np, enemy_img, cv.TM_CCOEFF_NORMED)
        result2 = cv.matchTemplate(img_np, enemy2_img, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        yloc, xloc = np.where(result >= 0.7)
        min_val2, max_val2, min_loc2, max_loc2 = cv.minMaxLoc(result2)
        yloc2, xloc2 = np.where(result2 >= 0.6)
        for (x, y) in zip(xloc, yloc):
            x_center, y_center = x + (w // 2), y + (h // 2)
            cv.rectangle(img_np,(x,y),(x+w,y+h),(0,255,0),1)
            print(x_center, y_center)
        for (x2, y2) in zip(xloc2, yloc2):
            x2_center, y2_center = x2 + (w2 // 2), y2 + (h2 // 2)
            cv.rectangle(img_np,(x2,y2),(x2+w2,y2+h2),(0,0,255),1)
            print(x2_center, y2_center)

        #mousex,mousey=pyg.position()
        #print('pos:',mousex,mousey)
        
        
        money_gray=cv.cvtColor(money_np,cv.COLOR_RGB2GRAY)
        exp_gray=cv.cvtColor(exp_np,cv.COLOR_BGR2GRAY)

        #cv.imshow("test", img_np)
        #cv.imshow("test2", money_gray)
        #cv.imshow('exp',exp_gray)

        
        extracted_money = pytesseract.image_to_string(money_np)
        extracted_exp = pytesseract.image_to_string(exp_gray)
        extracted_money = ''.join(char for char in extracted_money if char.isdigit())
        extracted_exp = ''.join(char for char in extracted_exp if char.isdigit())
        #cv.imwrite('video.png',img_np)
        print("Extracted Text:")
        #intMoney=int(extracted_money)
        print(extracted_money,' - ',extracted_exp)
        if extracted_money:
            if int(extracted_money)>=200 and flag1:
                buildTurret(2,1)
                flag1=False
            elif int(extracted_money)>=1200 and not flag1 and evoFlag1:
                buildTower()
                buildTurret(2,2)
                gecis=True
            elif int(extracted_money)>=1500 and flag2:
                sellTurret(2)
                sellTurret(1)
                buildTurret(2,2)
                buildTurret(2,1)
                flag2=False
            elif int(extracted_money)>=3000 and evoFlag2 and flag3:
                #sell turret
                sellTurret(1)
                buildTurret(2,1)
                flag3=False
            elif int(extracted_money)>=6000 and flagextra:
                sellTurret(2)
                #buildTower()
                buildTurret(3,2)
                flagextra=False
            elif int(extracted_money)>=6000  and flagextra2:
                sellTurret(1)
                buildTurret(3,1)
                flagextra2=False
            elif int(extracted_money)>=6000 and spot2:
                buildTower()
                spot2=False
            elif int(extracted_money)>=14000 and evoFlag3 and flag4:    
                #sellTurret(3)
                buildTurret(3,3)
                flag4=False
            elif int(extracted_money)>=14000  and flag5:
                sellTurret(1)
                buildTurret(3,1)
                flag5=False
            elif int(extracted_money)>=14000  and flag6:
                sellTurret(2)
                buildTurret(3,2)
                flag6=False
            elif int(extracted_money)>=7500 and spot3:
                buildTower()
                spot3=False
            elif int(extracted_money)>=14000 and flag7:
                buildTurret(3,4)
                flag7=False
            elif  int(extracted_money)>=100000 and flag8 and evoFlag4 :
                sellTurret(4)
                buildTurret(3,4)
                flag8=False


        if extracted_exp  :
            if int(extracted_exp)>=4000 and evoFlag1 and gecis:
                levelUp()
                evoFlag1=False
            elif int(extracted_exp)>=12000 and extraf:
                pyg.click(1109,238)
                extraf=False
            elif int(extracted_exp)>=14000 and not(evoFlag2):
                levelUp()
                evoFlag2=True
            elif int(extracted_exp)>=45000 and not(evoFlag3):
                levelUp()
                evoFlag3=True
                pyg.click(1109,238)
            elif int(extracted_exp)>=200000 and not(evoFlag4):
                levelUp()
                evoFlag4=True
        
        if(evoFlag2 and not(evoFlag3)):
            pyg.click(905,163,3)
        elif evoFlag1 and not(evoFlag2) and extracted_exp:
            if int(extracted_exp)<1500 and not(flag1):
                pyg.click(905,163)
            else:
                pyg.click(905,163)
                pyg.click(965,163,2) 
        elif evoFlag3 :
            pyg.click(905,163)
            if(extracted_money):
                if int(extracted_money)>=200000:
                    pyg.click(1085,163,2)
                    time.sleep(3)
                    pyg.click(1109,238)
        else:
            pyg.click(905,163)
            pyg.click(965,163,2)        
        cv.waitKey(0)
        cv.destroyAllWindows()
        time.sleep(space)
        #break
