from time import sleep
from json import dumps
import numpy as np
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x:
                         dumps(x).encode('utf-8'))


for e in range(1000):
    data = {'number' : np.random.rand(1,2)}

    #data = np.random.rand(0,1)
    producer.send('numtest', value=data)
    sleep(5)


