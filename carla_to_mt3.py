import json
import numpy as np

def carla_to_mt3(infile) -> None:
    
    data = []

    for line in infile:
        data.append(json.loads(line))
    
    #with open(r'jsonfile', 'r') as infile:
    #    for line in infile:
    #        data.append(json.loads(line))

    
    framesize=20
    batches=[]
    labels=[]
    unique_ids=[]
    labels_last_step=[]

    for k in range(2):
        
        batch=[]
        label=[]
        unique_id=[]

        randominteger=np.random.randint(1,(len(data)-framesize+1))
        counter=0

        basetime=data[randominteger-1][0].get('t')
        
        for n in range(framesize):
            counter += len(data[randominteger+n-1])

        for i in range(framesize):
            row = data[randominteger-1+i]
            for j in range(len(row)):
            ##Batch
                t=round(row[j].get('t')-basetime,3)
                r=row[j].get('r')
                vr=row[j].get('vr')
                phi=row[j].get('phi')

                batline=[r,vr,phi,t]
                batch.append(batline)

            ##Labels
                x=row[j].get('pointcloudx')
                y=row[j].get('pointcloudy')
                vx=row[j].get('vx')
                vy=row[j].get('vy')
                if row[j].get('id') > 0:
                    lab=row[j].get('id') 
                else:
                    lab=-1

                labline=[x,y,vx,vy,t,lab]
                label.append(labline)

            ##Annat
                unique_id.append(lab)
            
        batches.append(batch)
        labels.append(label) #+counter-len(data[randominteger+n])
        unique_ids.append(unique_id)

        a=[]

        for i in range(len(data[randominteger+n-1])):
            if labels[k][len(labels[k])-(len(data[randominteger+n-1]))+i][5] > 0:
                a.append(labels[k][len(labels[k])-(len(data[randominteger+n-1]))+i][0:4])
            else:
                continue
        labels_last_step.append(a)

    return batches, labels, unique_ids, labels_last_step
    
    #from carla_to_mt3 import carla_to_mt3
    #with open(r'jsonfile', 'r') as infile:
    #    a,b,c=carla_to_mt3.carla_to_mt3(infile)

#file_path = 'C:\\Users\\Emil\\Desktop\\P-lugg\\Chalmers\\GitChalmers\\Carla\\carla-radar-sim-trackformers\\hej\\hej.jsonl'
#with open(file_path, 'r') as infile:
    #batches, labels, unique_ids, labels_last_step  = carla_to_mt3(infile)
