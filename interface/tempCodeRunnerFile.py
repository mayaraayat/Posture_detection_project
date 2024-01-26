
@app.route('/record/<int:position>' , methods = ['GET'])
def record(position):
    if position == 0:
        empty_folder('recorded_positions')
        return None
    elif position == 8:
        return render_template('finish_record.html')

    filename = f'{position}.jpeg'
    update(position)
    return send_from_directory(app.config['POSITIONS'], filename)

def empty_folder(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Iterate through the files and remove them
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                # Recursively remove subdirectories
                empty_folder(file_path)
                os.rmdir(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

async def update(position):
    empty_folder('C:/Users/sotrd/Documents/Sensing Mat Demo Software/Recordings/')
    await asyncio.sleep(5)    
    filename1 = os.listdir('C:/Users/sotrd/Documents/Sensing Mat Demo Software/Recordings/')[0]
    shutil.move('C:/Users/sotrd/Documents/Sensing Mat Demo Software/Recordings/' + filename1, f'/Posture-detection/recorded_postions/{position}.json')
