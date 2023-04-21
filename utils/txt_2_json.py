import json

fields = ["instruction", "input", "output"]
# dict1 = {}
test = []

with open('unsupervised_train.txt') as f:
   count = 0
   for line in f:
    # print(line)
    if '<INPUT_START>' in line and '<INPUT_END>' in line:
      start = line.index('<INPUT_START>')
      end = line.index('<INPUT_END>') + 11
      
      dict2 = {}
      i = 0
      while i<len(fields):
        if i == 0:
          # creating dictionary for each employee
          dict2[fields[i]]= line[start:end]
        elif i == 1:
          dict2[fields[i]]= ""
        elif i == 2:
          dict2[fields[i]]= line
        i = i + 1
      
      # dict1[count]= dict2
      test.append(dict2)
      count += 1
      if count > 150000:
        break

out_file = open("llama_recipes.json", "w")
json.dump(test, out_file, indent = 4, sort_keys = False)
out_file.close()