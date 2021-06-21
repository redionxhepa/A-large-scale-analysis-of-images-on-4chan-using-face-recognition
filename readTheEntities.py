import ndjson
from optparse import OptionParser

#read the parsed argument
parser = OptionParser()
parser.add_option("-o","--input ",dest='input',help=" NDJSON file from 4chan/pol post",default="/home/redion/redion_files/Raiders_4Chan/pol_062016-112019_labeled.ndjson")
parser.add_option("-o","--output ",dest='output',help=" .txt file for output ",default="nameList_new.txt")
(options, arguments) = parser.parse_args()

#assigned the parsed argument
path_file= options.input
output_file =options.output


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
     

count=0
count_not=0
list_names={}
#go through each object
with open(path_file) as f:
    reader = ndjson.reader(f)
    for post in reader: 
      dates=ndjson_extract(post,"now")[0] # now of the first post   
      if(check_if_Between(dates)):  #check if it between July 2016 and July 2017
        #check the entities
        entity_label=ndjson_extract(post,"entity_label") 
        entity_text=ndjson_extract(post,"entity_text")
        # check the indeces of that are PERSON
        if(len(entity_label)>0):
          for index,value in enumerate(entity_label):
               if value=="PERSON":
                  person=entity_text[index]
                  if person in list_names:
                        list_names[person]=list_names[person]+1
                  else:
                        list_names[person]=1
# #write to the text file
outF = open(output_file, "w")
for name in list_names:
   # write line to output file
   outF.write(name + "   ---   "+ str(list_names[name]))
   outF.write("\n")
outF.close()



