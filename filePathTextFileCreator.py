import math
import os
from optparse import OptionParser
import pathlib

parser = OptionParser()
parser.add_option("-i", "--imagesFolderPath", dest='imagesfolder',
                  help=" Full path of the folder that containts all subdirectories of the images",
                  default='/home/redion/redion_files/facerecognition/WikiData/photosWiki')
parser.add_option("-n", "--txtFolderPath", dest='outputfolder', default="TextFilePathFolder",
                  help="Full path of the folder to save .txt file path of the images")
parser.add_option("-w", "--parts", dest='parts', default=2, help=" the number of .txt files to be created ")
(options, arguments) = parser.parse_args()

images_dir = options.imagesfolder
outputFolder = options.outputfolder
parts = int(options.parts)

# read the all the image paths on the all subdirectories
file_names = []
for root, dirs, files in os.walk(images_dir):
    for file in files:
        file_names.append('%s/%s' % (root, file))

print("Total number of images: " +str(len(file_names)))

# create the output folder if it does not exist
MYDIR = (outputFolder)
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    pathlib.Path(outputFolder).mkdir(parents=True, exist_ok=True)
    print("created folder : ", MYDIR)
else:
    print(MYDIR, "folder already exists.")

blockLength = math.floor(len(file_names) / parts)
print("Block length is: " +str(blockLength))

k = 0
for i in range(0, parts):
    with open(outputFolder + '/' + 'paths_' + str(i) + '.txt', 'a+') as f:
        while True:
            f.write(file_names[k])
            f.write('\n')
            k = k + 1
            if (k >= len(file_names) or k % blockLength == 0):
                break
print("Creation of text files is done")

