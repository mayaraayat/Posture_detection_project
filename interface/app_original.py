
import os
import numpy as np
#import traitement as tr
from flask import Flask, flash, request, redirect, url_for, render_template,send_file, send_from_directory
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from io import BytesIO
import shutil
from flask_socketio import SocketIO
string = '{"softwareVersion":"1.9","calibrationConfiguration":{"calibrationTable":{"conversionTable":[{"digitalValue":0.0,"mmHgValue":0.0,"psiValue":0.0,"gcm2":0.0},{"digitalValue":9.0,"mmHgValue":9.0,"psiValue":0.1740312,"gcm2":12.23559},{"digitalValue":40.0,"mmHgValue":19.0,"psiValue":0.3673992,"gcm2":25.83069},{"digitalValue":197.0,"mmHgValue":28.0,"psiValue":0.5414304,"gcm2":38.06628},{"digitalValue":675.0,"mmHgValue":37.0,"psiValue":0.7154616,"gcm2":50.30187},{"digitalValue":895.0,"mmHgValue":47.0,"psiValue":0.9088296,"gcm2":63.89697},{"digitalValue":1250.0,"mmHgValue":56.0,"psiValue":1.082861,"gcm2":76.13256},{"digitalValue":1540.0,"mmHgValue":66.0,"psiValue":1.276229,"gcm2":89.72766},{"digitalValue":1628.0,"mmHgValue":75.0,"psiValue":1.45026,"gcm2":101.963249},{"digitalValue":2115.0,"mmHgValue":84.0,"psiValue":1.624291,"gcm2":114.198837},{"digitalValue":2145.0,"mmHgValue":94.0,"psiValue":1.817659,"gcm2":127.793938},{"digitalValue":2518.0,"mmHgValue":112.0,"psiValue":2.165721,"gcm2":152.793945},{"digitalValue":2745.0,"mmHgValue":131.0,"psiValue":2.533121,"gcm2":178.09581},{"digitalValue":2774.0,"mmHgValue":150.0,"psiValue":2.90052,"gcm2":203.9265},{"digitalValue":2985.0,"mmHgValue":169.0,"psiValue":3.267919,"gcm2":229.757187},{"digitalValue":3124.0,"mmHgValue":187.0,"psiValue":3.615981,"gcm2":254.228363},{"digitalValue":3131.0,"mmHgValue":206.0,"psiValue":3.983381,"gcm2":280.059052},{"digitalValue":3133.0,"mmHgValue":225.0,"psiValue":4.35078,"gcm2":305.88974},{"digitalValue":3300.0,"mmHgValue":244.0,"psiValue":4.718179,"gcm2":331.720428},{"digitalValue":3310.0,"mmHgValue":262.0,"psiValue":5.066241,"gcm2":356.19162},{"digitalValue":3321.0,"mmHgValue":281.0,"psiValue":5.43364,"gcm2":382.0223},{"digitalValue":3456.0,"mmHgValue":332.0,"psiValue":6.419817,"gcm2":451.35733},{"digitalValue":3549.0,"mmHgValue":361.0,"psiValue":6.980585,"gcm2":490.7831},{"digitalValue":3627.0,"mmHgValue":398.0,"psiValue":7.696046,"gcm2":541.3573},{"digitalValue":3673.0,"mmHgValue":417.0,"psiValue":8.063445,"gcm2":566.915649},{"digitalValue":3683.0,"mmHgValue":454.0,"psiValue":8.778907,"gcm2":617.2175},{"digitalValue":3736.0,"mmHgValue":492.0,"psiValue":9.513705,"gcm2":668.8789},{"digitalValue":3819.0,"mmHgValue":529.0,"psiValue":10.22917,"gcm2":719.1808},{"digitalValue":4095.0,"mmHgValue":550.0,"psiValue":10.63524,"gcm2":747.7305}],"name":null},"customCalibrationTable":null,"calibrationUnits":0,"isVoltageSharingCompensationActivated":true,"isOutputImpedanceCompensationActivated":true,"isAutomaticBrokenRowsCompensationActivated":false,"isRoundMatrixFilterActivated":true,"isManualRowInterpolationActivated":[false,false,false,false,false,false,false,false,false],"manualRowInterpolationValue":[0,0,0,0,0,0,0,0,0],"isManualColumnInterpolationActivated":[false,false,false],"manualColumnInterpolationValue":[0,0,0]},"representationConfiguration":{"threshold":100,"scaleSetting":0,"automaticScaleDefinition":0,"manualScaleDefinition":0,"manualScaleMax":50,"isContourActivated":true,"rotateDegrees":0,"flip":0,"plotBackgroundColor":{"A":255,"B":255,"G":255,"R":255},"colorScale":{"Colors":[{"A":255,"B":139,"G":0,"R":0},{"A":255,"B":190,"G":113,"R":0},{"A":255,"B":242,"G":226,"R":0},{"A":255,"B":170,"G":255,"R":84},{"A":255,"B":56,"G":255,"R":198},{"A":255,"B":0,"G":235,"R":255},{"A":255,"B":0,"G":195,"R":255},{"A":255,"B":0,"G":146,"R":242},{"A":255,"B":0,"G":73,"R":190},{"A":255,"B":0,"G":0,"R":139}]},"isDiscardBorderActivated":false,"numDiscardedBorderLines":1,"isShowNumbersActivated":false},"pressureData":['


if not os.path.exists('./static/'):
        os.mkdir('./static/')
app = Flask(__name__ , static_url_path='/static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

image_folder = os.path.join('static','uploads')

image_folder1 = os.path.join('static','uploads1')


static_folder= os.path.join('static','images')
app.config['Gal_FOLDER'] = static_folder

app.config['IMAGE_FOLDER'] = image_folder





 
UPLOAD_FOLDER = 'static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'




@app.route('/')
def home():
    return render_template('home.html')
# Index
@app.route('/Contact')
def about():
    return render_template('contact.html')

@app.route('/posture')
def posture():
    return render_template('posture.html')



shutil.move('C:/Users/sotrd/Documents/Sensing Mat Demo Software/Recordings/SensingMatData_231215_084647.json', 'C:/Users/sotrd/Desktop/Interface/Posture-detection/data.json')


with open("data.json", 'r') as file:
    content = file.read()




# Suppression de la virgule à la fin du fichier (si elle existe)
content = content.rstrip(',\n')  # Retire la virgule ou le saut de ligne en fin de fichier

# Ajout de la fermeture du dictionnaire
content += ']}'

content = string + content

# Sauvegarde du fichier
with open('data.json', 'w') as file:
    file.write(content)


# Load data from the JSON file
with open("data.json") as f:
    data = json.load(f)
    pressure_data = data.get('pressureData', [])
    pressure_matrices = [entry["pressureMatrix"] for entry in pressure_data[-20:]]

for i, matrix in enumerate(pressure_matrices):
    rotated_matrix = np.rot90(matrix, 0)  # Rotate the matrix by 180 degrees
    fig, ax = plt.subplots()
    im = ax.imshow(rotated_matrix, cmap='coolwarm')
    #im = ax.imshow(pressure_matrices, cmap='hot')
    plt.axis('off')  # Hide axis for cleaner images
    plt.savefig(os.path.join(image_folder, f'frame_{i}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()




c = 0

@app.route('/get_frame/<int:frame>')
def get_frame(frame):
    global c
    if frame == 0 and c >= 1:
                
        shutil.move('C:/Users/sotrd/Documents/Sensing Mat Demo Software/Recordings/SensingMatData_231215_084647.json', 'C:/Users/sotrd/Desktop/Interface/Posture-detection/data.json')


        with open("data.json", 'r') as file:
            content = file.read()

        # Suppression de la virgule à la fin du fichier (si elle existe)
        content = content.rstrip(',\n')  # Retire la virgule ou le saut de ligne en fin de fichier

        # Ajout de la fermeture du dictionnaire
        content += ']}'

        content = string + content

        # Sauvegarde du fichier
        with open('data.json', 'w') as file:
            file.write(content)


        # Load data from the JSON file
        with open("data.json") as f:
            data = json.load(f)
            pressure_data = data.get('pressureData', [])
            pressure_matrices = [entry["pressureMatrix"] for entry in pressure_data[-20:]]

        for i, matrix in enumerate(pressure_matrices):
            rotated_matrix = np.rot90(matrix, 0)  # Rotate the matrix by 180 degrees
            fig, ax = plt.subplots()
            im = ax.imshow(rotated_matrix, cmap='coolwarm')
            #im = ax.imshow(pressure_matrices, cmap='hot')
            plt.axis('off')  # Hide axis for cleaner images
            plt.savefig(os.path.join(image_folder, f'frame_{i}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()
        c = 0
    if frame == 0:
        c += 1
    filename = f'frame_{frame}.png'

    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

    


if __name__ == '__main__':
    app.run(debug = True)
