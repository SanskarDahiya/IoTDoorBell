import cloudinary
import cloudinary.uploader
import cloudinary.api

from credentials import cloudinary as cld

cloudinary.config( 
  cloud_name = cld['name'], 
  api_key = cld['key'], 
  api_secret = cld['secret'] 
)


def UploadPhoto(path):
    x = cloudinary.uploader.upload(path)
    print('Photo Uploaded')
    return x['url']