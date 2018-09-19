#Importacion de paquetes
import numpy as np
import argparse
import imutils
from imutils import contours
import cv2 as cv
import time

def image_processing():
    #Definicion de los argumentos para el funcionamiento del programa
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("-i","--image",required=True,help="Path de la imagen")
    argumentParser.add_argument("-r","--reference",required=True,help="Path de la imagen de referencia")
    args=vars(argumentParser.parse_args())

    #Lectura de la imagen de referencia
    ref=cv.imread(args["reference"],0)
    ref=cv.threshold(ref, 10, 255, cv.THRESH_BINARY_INV)[1] #Se invierte el contraste binario de la imagen, es decir, fondo negro y resalte blanco

    #Reconocimiento de contornos
    refCountour = cv.findContours(ref.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) #Se realiza una copia superficial de la imagen. Se define una jerarquia externa. Y se realiza una aproximacion simple para ahorrar memoria
    if imutils.is_cv2(): #Verificacion de la version del OpenCV
        refCountour = refCountour[0]
    else:
        refCountour = refCountour[1]
    refCountour = contours.sort_contours(refCountour, method="left-to-right")[0]#Se ordena los contornos de izq a der
    refDigits={}

    #Extraccion de digitos
    for ( di, co) in enumerate(refCountour):
        (x, y, w, h) = cv.boundingRect(co) #Se selecciona un area rectangular sin rotacion debido a que sabemos como es la forma a extraer de ante mano
        roi = ref[y-1:y+h+1, x-1:x+w+1] #Se define la region de interes de acuerdo al retorno de la funcion anterior
        roi = cv.resize(roi, (57,88))#Se necesita un valor fijo para poder hacer matching luego
        refDigits[di]=roi

    #Estructuras para transformacion
    rectKernel = cv.getStructuringElement(cv.MORPH_RECT,(9,3))#Matriz rectangular
    squKernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))#Matriz cuadrada
    smKernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))

    #Lectura de imagen
    image=cv.imread(args["image"],0)
    image=imutils.resize(image,width=300)
    tophat=cv.morphologyEx(image, cv.MORPH_TOPHAT, rectKernel)#Transformacion Tophat
    #Transformacion gradiente hacia X
    gradX=cv.Sobel(tophat,ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)#Se usa Sobel para poder sacar los bordes de una imagen complicada
    gradX = np.absolute(gradX)#Se lleva a
    (valMin, valMax) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - valMin)/(valMax - valMin)))
    gradX = gradX.astype("uint8")#Como se uso cv32 el cual es flotante, se tiene que devolver a uint8 para poder realizar operaciones sobre el
    #Transformacion cerrado
    gradX=cv.morphologyEx(gradX, cv.MORPH_CLOSE, rectKernel)
    thresh=cv.threshold(gradX,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]


    #Hallazgo de contornos
    imgContour = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if imutils.is_cv2(): #Verificacion de la version del OpenCV
        imgContour = imgContour[0]
    else:
        imgContour = imgContour[1]
    imgDigits=[]
    #Extraccion de digitos
    for ( di, co) in enumerate(imgContour):
        (x, y, w, h) = cv.boundingRect(co)
        ar=w/float(h)#Aspect ratio de los contornos
        if ar>5 and ar<8.8: #Estos valores se sacaron experimentalmente a prueba y error, porque no habia forma de identificar cual contorno es cual a simple vista
            if ((w>101 and w<137)and(h>11.5 and h<20)):#Igual estos
                imgDigits.append((x, y, w, h))
        #s = ('Width: '+repr(w)+' Height: '+repr(h)+' Ratio: '+repr(ar))
        #print s
    imgDigits = sorted(imgDigits, key=lambda x:x[0])
    output = []
    #Ciclo sobre los grupos seleccionados
    for (i,(gX, gY, gW, gH)) in enumerate(imgDigits):#Se hace un ciclo para dado el caso de que debido a las morfologias que se le aplican a la imagen quede dividida la cedula en varios grupos
        groupOutput = []
        group=image[gY-5:gY+gH+5 , gX-5:gX+gW+5]#Los -5 y +5 es para darle un espacio a los bordes y no quede exactamente desde el inicio el area de interes
        group = cv.morphologyEx(group, cv.MORPH_TOPHAT, squKernel)#Transformacion Tophat
        group = cv.threshold(group,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1] #Binarizacion con otsu para mejorar la binarizacion

        digitContour = cv.findContours(group.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if imutils.is_cv2(): #Verificacion de la version del OpenCV
            digitContour = digitContour[0]
        else:
            digitContour = digitContour[1]
        digitContour=contours.sort_contours(digitContour, method="left-to-right")[0]
        #Hallazgo de cada digito
        for c in digitContour:
            (x, y ,w ,h) = cv.boundingRect(c) #Bordes para los digitos en los grupos
            roi = group[y:y+h, x:x+w]
            roi = cv.resize(roi, (57,88))

            scores=[] #Puntajes del matching
            for (digit, digitROI) in refDigits.items():
                result = cv.matchTemplate(roi, digitROI, cv.TM_CCOEFF)
                (_,score,_,_)=cv.minMaxLoc(result)
                scores.append(score)

            groupOutput.append(str(np.argmax(scores)))
            cv.rectangle(image, (gX-5, gY-5), (gX+gW+5, gY+gH+5), (0,0,255),2)
            cv.putText(image,"".join(groupOutput), (gX,gY-15),cv.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255,2))
            output.extend(groupOutput)

    cv.imshow('imagen', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    for y in range(0,10):
        s = ('CC: '+repr(groupOutput[y]))
        print(s)

def main():
    image_processing()

main()