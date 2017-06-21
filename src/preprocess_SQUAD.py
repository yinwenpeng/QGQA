import json
from pprint import pprint

path='/mounts/data/proj/wenpeng/Dataset/SQuAD/'

def  parse_train():
#     read_file=open(path+'train-v1.0.json', 'r')
    with open(path+'train-v1.0.json') as data_file:    
        data = json.load(data_file)

#     pprint(data['data'][0]['paragraphs'][0])
    doc_size=len(data['data'])
    print 'doc_size:', doc_size
    para_size=0
    for i in range(doc_size):
#         pprint(data['data'][i]['paragraphs'])
#         print 
#         exit(0)
        para_size_i=len(data['data'][i]['paragraphs'])
        print 'para_size_i:', para_size_i
        para_size+=para_size_i
#     pprint(len(data['data']))
#     print data['data'][0]['paragraphs'][0]
    print para_size

if __name__ == '__main__':
    parse_train()