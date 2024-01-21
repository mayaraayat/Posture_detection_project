import cv2
import pickle
import keyboard
import string
import numpy as np
# insérer la longueur et la largeur de chacun des rectangles délimitant les places du parking
rectW = 55
rectH = 120

try:
    with open('chairpos', 'rb') as f:
        poslist = pickle.load(f)
except:
    poslist = []  # on définit une liste des différentes positions des rectangles qui délimitent les places du parking
L = [] 
def mouseClick(events, x, y, flags, params):
    global poslist
    global L
    if events == cv2.EVENT_LBUTTONDOWN:  # si l'évènement de la souris est un clic gauche
        L.append((x,y))  # on ajoute le point cliqué à la liste L
        print(L)
        if len(L) == 4:
            # on ajoute le couple définissant la position du rectangle à la liste
            poslist.append(L)
            L.clear
    if events == cv2.EVENT_RBUTTONDOWN:  # si l'évènement de la souris est un clic droit
        for i, pos in enumerate(poslist):
            quad_array = np.array(pos, dtype=np.int32).reshape((-1, 1, 2))

            # Check if the point is inside the quadrilateral
            if cv2.pointPolygonTest(quad_array, (x,y), False) >= 0:
                poslist.pop(i)
    with open('chairpos', 'wb') as f:
        pickle.dump(poslist, f)


def test(imgurl):
    '''# tant qu'on ne clique pas sur echap, l'image reste ouverte.
    while keyboard.is_pressed("x") == False:
        img = cv2.imread(imgurl)
        # k = len(poslist)//n  # nb_de_lignes
        for index, pos in enumerate(poslist):
            quad_array = np.array(pos, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img,[quad_array],True,(255,255,255),thickness=2)

        cv2.imshow('image', img)
    # cette fonction sert à gérer les évènements qui accompagnent un clic, notamment dans notre cas un ajout ou une suppression d'un couple de la liste.
        cv2.setMouseCallback('image', mouseClick)
        cv2.waitKey(0)
    cv2.imwrite('./static/images/parking_spots.jpg',img)'''
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouseClick)

    while True:
        img = cv2.imread(imgurl)
        for index, pos in enumerate(poslist):
            quad_array = np.array(pos, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [quad_array], True, (255, 255, 255), thickness=2)

        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('x'):  # Press 'x' to exit the loop
            print(poslist)
            break

    cv2.imwrite('./static/images/parking_spots.jpg', img)
    cv2.destroyAllWindows()



if __name__== '__main__':
    test('./static/images/chair01.jpg')
    # print(poslist)
    #poslist = []

    '''def mouseClick(events, x, y, flags, params):
        global poslist
        if events == cv2.EVENT_LBUTTONDOWN:
            poslist.append((x, y))
            with open('carparkposition', 'wb') as f:
                pickle.dump(poslist, f)

            img = cv2.imread('./static/images/chair01.jpg')
            for point in poslist:
                cv2.circle(img, point, 3, (255, 255, 255), -1)

            cv2.imshow('image', img)

    # Load the image
    img = cv2.imread('./static/images/chair01.jpg')
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouseClick)

    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cv2.destroyAllWindows()'''