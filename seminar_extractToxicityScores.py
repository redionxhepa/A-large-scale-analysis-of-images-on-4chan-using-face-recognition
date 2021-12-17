import ndjson
import json
import sys
from optparse import OptionParser
from tqdm import tqdm
#read the parsed argument
parser = OptionParser()
parser.add_option("-o","--input ",dest='input',help=" NDJSON file from 4chan/pol post",default="/home/redion/redion_files/Raiders_4Chan/pol_062016-112019_labeled.ndjson")
(options, arguments) = parser.parse_args()

#assigned the parsed argument
path_file= options.input


#helper function to extract a particular key value from a ndjson file
def ndjson_extract(obj, key):
    """Recursively fetch values from nested NDJSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in NDJSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    return values


#helper function to chekc if the input date is between a particular range (06/30/16(Thu)12:39:23 (Example input))
def check_if_Between(input_data):
     month_input=int(input_data[0:2]) #it will take care on its own about the 0 at the fronf if there is one
     year_input=int(input_data[6:8])  
     if (year_input==16 or year_input==17):
       if (year_input==16 and month_input>=7):
              return True
       if(year_input==17 and month_input<=7):
              return True
     return False  
     


#read the md5-s  of the clustered images
clustered_imagesMD5=[]
with open('md5OFClusteredImages.txt','r') as file_zeta:
    for line_z in file_zeta:
        clustered_imagesMD5.append(line_z.strip())

clustered_imagesMD5_SET=set(clustered_imagesMD5)
print(len(clustered_imagesMD5))
print(len(clustered_imagesMD5_SET))

print("---")
#counters for posts
totalThreads=0
postsWithBothText_Photo=0
postsWithOnlyText=0
postsWithOnlyPhoto=0
postsTotal=0
countTask=0

#counters for toxicitiy
highToxicityTextWithPhoto=0
highToxicityTextWithoutPhoto=0

#md5-s of images  in the high toxicity posts
md5_highToxicity=[]
# severe toxicity threshold
severe_toxicity_threshold=0.7
count=0


relevant_posts=[]

#relevant dataset 
#go through each object
with open(path_file) as f:
    reader = ndjson.reader(f)
    for _,thread in enumerate(tqdm(reader)):
        totalThreads=totalThreads+1
        for post in thread["posts"]:
              postsTotal=postsTotal+1
              
              #check if the
              if "md5" in post :
                   if post["md5"] in clustered_imagesMD5_SET:
                        relevant_posts.append(post)
              if "md5" in post and post['perspectives'] : # it has a photo and text
                  postsWithBothText_Photo = postsWithBothText_Photo+1
                  if  post['perspectives']["SEVERE_TOXICITY"] >= severe_toxicity_threshold:
                        highToxicityTextWithPhoto=highToxicityTextWithPhoto+1
                        md5_highToxicity.append(post["md5"])
              elif "md5" in post :   #it has only  photo
                  postsWithOnlyPhoto=postsWithOnlyPhoto+1
              elif  post['perspectives'] :   #it has only text
                  postsWithOnlyText=postsWithOnlyText+1
                  if  post['perspectives']["SEVERE_TOXICITY"] >= severe_toxicity_threshold:
                        highToxicityTextWithoutPhoto=highToxicityTextWithoutPhoto+1  
  

       



# relevant posts
print(len(relevant_posts))
print(relevant_posts[0])


with open("relevantPostsNewWrite.json","a") as f:
    for relevenatPost in  relevant_posts:
              f.write(json.dumps(relevenatPost) + '\n') 



#with open ("relevantPostsNewWrite.json","r") as f_2:
#       for line_2 in f_2:
#        #read each line of the json file and extract the information
#        current_data = json.loads(line_2)
#        print(current_data)
#        break
