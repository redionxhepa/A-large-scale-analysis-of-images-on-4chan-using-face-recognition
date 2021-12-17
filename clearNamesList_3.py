from PIL import Image
import requests
from io import BytesIO
import numpy as np
from wikidata.client import Client
print("Clearing names")

#global variables and counters
threshold=5
count=0
count_initial=0
length_name_threshold=3  # at least 3     


#this is a helper function to  find a key in a dictionary otherwise turn None

def search(d, key, default=None):
    """Return a value corresponding to the specified key in the (possibly
    nested) dictionary d. If there is no item with that key, return
    default.
    """
    stack = [iter(d.items())]
    while stack:
        for k, v in stack[-1]:
            if isinstance(v, dict):
                stack.append(iter(v.items()))
                break
            elif k == key:
                return v
        else:
            stack.pop()
    return default



names=[]
with open('/home/redion/redion_files/facerecognition/WikiData/list_edited_'+str(threshold)+'length_name_atLeast_threshold'+str(length_name_threshold)+'.txt',"w")as file:
 with open('nameList_new.txt') as file2:
    for line in file2:
        count_initial=count_initial+1
        tokens=line.strip().split()
        if(len(tokens)<3):
          continue
        try:
           name_count=int(tokens[-1])  # this might be redundant if we only check Name Surname  type of entities
        except:
           continue

        #check if the name is repeated enough times
        if(name_count<threshold):
               continue
        #now we need to remove  "----","int number count"     
        tokens=tokens[:-2]
        if(len(tokens)>=length_name_threshold):
          name=" ".join(tokens) 
          names.append(name)
          count= count+1
          #write the new name in the file
          file.write(name)
          file.write("\n") 
          if count<20:
            # break
            print(tokens)
file2.close()
file.close()
print("Threshold is:  " + str(threshold))
print("final number of names after edit") 
print(count)
#print(len(names))

print("Number of names initially (unedited)")
print(count_initial)

print("Name Editing Ended")

#prepare the code to get the WikiData ID for any particular user
S = requests.Session()
URL = "https://en.wikipedia.org/w/api.php"
have_entry=[]
counter_haveEntry=0
counter2=0
counter_names=0
#try to get the url  from the names wikidata
for name in names:
  
  # print("One name")
  # print(name)
  # break
   counter_names=counter_names+1
   if(counter_names % 50==0):
     print("Counter of all names " + str(counter_names)) 
   client = Client()
   #get the id
   PARAMS = {
    "action": "query",
    "titles": name,
    "prop": "pageprops",
    "format": "json"
   }
   R = S.get(url=URL, params=PARAMS)
   DATA = R.json()  
   id_user=search(DATA,'wikibase_item')
   if (id_user is  None):
       continue
  #get the url
   try:
    counter_haveEntry=counter_haveEntry+1
    have_entry.append(name)
    entity = client.get(id_user, load=True)
    image_prop = client.get('P18')
    image = entity[image_prop]
    url=image.image_url
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save("/home/redion/redion_files/facerecognition/WikiData/photosMulti/"+name+".png")
    counter2=counter2+1
   except:
    continue
  #face_encoding=face_recognition.face_encodings(image_last,model="cnn")

print("Names with photo")
print(counter2)
print("Downloading the photos ended")
print("Have wiki entry")
print(counter_haveEntry)


file3=open("/home/redion/redion_files/facerecognition/WikiData/multiNames_haveEntry.txt","w")
for entry in have_entry:
      file3.write(entry)
      file3.write("\n")

file3.close()
