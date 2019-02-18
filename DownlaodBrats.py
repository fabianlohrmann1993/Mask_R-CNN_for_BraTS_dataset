'''
Created on Jun 17, 2018
@author: daniel
'''
import requests, zipfile, os, sys
print(sys.version_info)
if sys.version_info<(3,0,0):
    import StringIO
else:
    import io
if not os.path.exists("Data/"):
    os.makedirs("Data/")
    os.makedirs("Data/BRATS_2018")
    url = "https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/2018/MICCAI_BraTS_2018_Data_Training.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("Data/BRATS_2018")