
# 1. save photo with name using opencv
# 2. upload photo to server

from findFace import *
from uploadToCloudinary import UploadPhoto
from sendMessage import sendSMS

print('*****************************START*******************************')
while True:
    z = input('Press 1 to Capture: \n')
    if(z!='1'):
        break
    [path,username] = Capture_Face()
    message = "Hello, There is someone at the door.\n"
    message += "Name:"
    if(username):
        message+=username
    else:
        message+=" Not in Database"

    ImgUrl = UploadPhoto(path)

    message += "\nHere is link of photo "+ImgUrl

    print(message)
    sendSMS(message,8059111155)
print('*****************************EXIT********************************')