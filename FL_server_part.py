import hashlib
import http
import json
import threading
import time
from urllib.parse import urlparse
from uuid import uuid4
import numpy as np
import winerror

import requests
from flask import Flask, jsonify, request


class Federated_learning_server:
    def __init__(self):
        #federated learning
        self.client_adresses = []
        
        self.mixedPrecision = False
        self.scaledWeightsSize = 1
        keywords_buttons = {
            "montserrat": 1,
            "pedraforca": 2,
            "vermell": 3,
            # "blau": 4,
            # "verd": 5
        }
        self.numdevices = 3
        self.num_devices_sent = 0
        input_nodes = 650
        output_nodes = len(keywords_buttons)
        size_hidden_nodes = 25
        self.size_hidden_layer = (input_nodes+1)*size_hidden_nodes
        self.size_output_layer = (size_hidden_nodes+1)*output_nodes
        self.devices_hidden_layer = np.empty((self.numdevices, self.size_hidden_layer), dtype='float32')
        self.devices_output_layer = np.empty((self.numdevices, self.size_output_layer), dtype='float32')
        
        self.received_min_w = np.empty((self.numdevices), dtype='float32')
        self.received_max_w = np.empty((self.numdevices), dtype='float32')

        self.devices_hidden_layer_batch_amount = np.zeros((self.numdevices))

        self.devices_num_epochs = []

# Instantiate the Node
app = Flask(__name__)

# Generate a globally unique address for this node
node_identifier = str(uuid4()).replace('-', '')

# Instantiate the server
federated_learning_server = Federated_learning_server()


#functions for my Portenta Project
@app.route('/test/post', methods=['POST'])
def test_post():
    values = request.get_json()
    
    print(request.headers)
    print(request.cookies)
    print(request.data)
    print(request.args)
    print(request.form)
    print(request.endpoint)
    print(request.method)
    print(request.remote_addr)
    response = {
        'message': 'POST message received',
        'content': values,
    }
    return jsonify(response), 201

@app.route('/test/AsyncWebServer', methods=['POST'])
def test_AsyncWebServer():
    values = request.get_json()
    
    test = values.get('message')
    print(test)
    response = {
        'message': 'AsyncWebServer message received',
        # 'content': values,
    }
    return jsonify(response), 201

#get the IP of Portenta boards
@app.route('/register', methods=['GET'])
def register_board():
    adress = request.remote_addr
    federated_learning_server.client_adresses.append(adress)

    response = {
        'ip': adress
    }
    return jsonify(response), 201

def deScaleWeight(min_w, max_w, weight):
    a, b = getScaleRange()
    return min_w + ( (weight-a)*(max_w-min_w) / (b-a) )

def scaleWeight(min_w, max_w, weight):
    a, b = getScaleRange()
    return round(a + ( (weight-min_w)*(b-a) / (max_w-min_w) ))

def getScaleRange():
    if federated_learning_server.scaledWeightsSize == 1:
        return -128, 127
    elif federated_learning_server.scaledWeightsSize == 2:
        return -32768, 32767
    elif federated_learning_server.scaledWeightsSize == 4:
        return -2147483648, 2147483647

#get the model of a board
@app.route('/FL/sendData', methods=['POST'])
def FlGetModelData():
    #check if the board has contacted before, and if not add its address to the list
    adress = request.remote_addr
    if not (adress in federated_learning_server.client_adresses):
        #the adress isn't in the list, so we add it
        federated_learning_server.client_adresses.append(adress)
    
    #get device index from the list of adresses
    device_index = federated_learning_server.client_adresses.index(request.remote_addr)
    values = request.get_json()
    
    num_epochs = values.get('num_epochs') #int
    print(f"Receiving model from device {adress}")
    # print("num_epochs: ")
    # print(num_epochs)
    print(f"num_epochs: {num_epochs}")
    federated_learning_server.devices_num_epochs.append(num_epochs)

    min_w = values.get('min_weight') #float
    max_w = values.get('max_weight') #float
 

    federated_learning_server.received_min_w[device_index] = min_w
    federated_learning_server.received_max_w[device_index] = max_w    
    # print("min_weight: ")
    # print(min_w)
    print(f"min_weight: {min_w}")
    print(federated_learning_server.received_min_w[device_index])
    # print("max_weight: ")
    # print(max_w)
    print(f"max_weight: {max_w}")
    print(federated_learning_server.received_max_w[device_index])

    response = {
        'message': 'model data received from device: ',
        'device' : device_index
    }
    return jsonify(response), 201


#get the model of a board
@app.route('/FL/sendHiddenNodeBatch', methods=['POST'])
def FlGetHiddenNodeBatch():
    #check if the board has contacted before, and if not add its address to the list
    adress = request.remote_addr
    if not (adress in federated_learning_server.client_adresses):
        #the adress isn't in the list, so we add it
        federated_learning_server.client_adresses.append(adress)
    
    #get device index from the list of adresses
    device_index = federated_learning_server.client_adresses.index(request.remote_addr)
    values = request.get_json()

    min_w = federated_learning_server.received_min_w[device_index]
    max_w = federated_learning_server.received_min_w[device_index]

    #increment the counter of batches received 
    batchCounter = federated_learning_server.devices_hidden_layer_batch_amount[device_index] + 1
    federated_learning_server.devices_hidden_layer_batch_amount[device_index] = batchCounter

    a, b = getScaleRange()

    batchSize = values.get('batchSize')
    batchNumber = values.get('batchNumber')
    lastInBatch = values.get('lastInBatch')
    numBatches = values.get('numBatches')

    print(f"Receiving hidden weight batch from device {adress}")
    print(f"Batch number: {batchNumber}")
    # print("batch: ")
    # print(batchNumber)

    #  Receiving model...
    # print("HiddenWeights: ")
    received_hidden_layer = values.get('HiddenWeights')
    # print(received_hidden_layer)
    for i in range(lastInBatch): # hidden layer
        if federated_learning_server.mixedPrecision:
            scaledWeight = received_hidden_layer[i] #int
            #scaled_in_float = readFloat(d)
            # float_weight = readFloat(d)
            weight = deScaleWeight(min_w, max_w, scaledWeight)
            # if i < 5 and d.port == 'com6': print(f"[{d.port}] Recevied Weight {i}: {float_weight}")
            # if abs(float_weight - weight) > 0.3: print(f"[{d.port}] Scaled weight: {scaledWeight} (float: {scaled_in_float}), Float weight (hid): {float_weight}, descaled: {weight}. Difference: {abs(float_weight - weight)}")
        else: weight = received_hidden_layer[i] #float
        
        # Received Weight {i}: {weight}")
        federated_learning_server.devices_hidden_layer[device_index][i+batchNumber*batchSize] = weight

    #when the server receives a model calculate the medium of all previously received models
    # print("lastInBatch: ")
    # print(lastInBatch)
    # print("batchSize: ")
    # print(batchSize)
    # print(lastInBatch != batchSize)
    
    #checking if the whole model is sent using the last batch only works if the messages arrive correctly and ordered
    # if the last batch reaches the server before other batches they will be lost 
    if (lastInBatch != batchSize):
        print("It's the last batch")
        # # the whole model is received
        # federated_learning_server.num_devices_sent = federated_learning_server.num_devices_sent+1
        # # if all devices have sent a model start FL
        # if (federated_learning_server.num_devices_sent == federated_learning_server.numdevices):
        #     federated_learning_server.num_devices_sent = 0
        #     doFL()
    else:
        print("It's not the last batch")

    # using the amount of batches received adds an element to the message to get the total amount,
    # but will work no matter the order of the batches.
    if (batchCounter == numBatches):
        # the whole model is received
        federated_learning_server.num_devices_sent = federated_learning_server.num_devices_sent+1
        # if all devices have sent a model start FL
        if (federated_learning_server.num_devices_sent == federated_learning_server.numdevices):
            federated_learning_server.num_devices_sent = 0
            doFL()
    
    response = {
        'message': 'Hidden Node Batch received from device: ',
        'device' : device_index
    }
    return jsonify(response), 201

#get the model of a board
@app.route('/FL/sendOutputNodes', methods=['POST'])
def FlGetOutputNodes():
    #check if the board has contacted before, and if not add its address to the list
    adress = request.remote_addr
    if not (adress in federated_learning_server.client_adresses):
        #the adress isn't in the list, so we add it
        federated_learning_server.client_adresses.append(adress)
    
    #get device index from the list of adresses
    device_index = federated_learning_server.client_adresses.index(request.remote_addr)
    values = request.get_json()

    min_w = federated_learning_server.received_min_w[device_index]
    max_w = federated_learning_server.received_max_w[device_index]
    
    print(f"Receiving output weights from device {adress}")
    # print("OutputWeights: ")
    received_output_layer = values.get('OutputWeights')
    # print(received_output_layer)
    
    # print(len(received_output_layer))
    # print(federated_learning_server.size_output_layer)
    for i in range(federated_learning_server.size_output_layer): # output layer
        if federated_learning_server.mixedPrecision:
            scaledWeight = received_output_layer[i] #int
            #scaled_in_float = readFloat(d)
            # float_weight = readFloat(d)
            weight = deScaleWeight(min_w, max_w, scaledWeight)
            #if abs(float_weight - weight) > 0.3: print(f"[{d.port}] Scaled weight: {scaledWeight} (float: {scaled_in_float}), Float weight (hid): {float_weight}, descaled: {weight}. Difference: {abs(float_weight - weight)}")
        else: weight = received_output_layer[i] #float
        
        federated_learning_server.devices_output_layer[device_index][i] = weight
        # print(i)
        # print(received_output_layer[i])
        # print(federated_learning_server.devices_output_layer[device_index][i])

    # # if it was not connected before, we dont use the devices' model
    # if not d in old_devices_connected:
    #     devices_num_epochs[device_index] = 0
    #     print(f"[{d.port}] Model not used. The device has an outdated model")


    #when the server receives a model calculate the medium of all previously received models

    response = {
        'message': 'model received from device: ',
        'device' : device_index
    }
    return jsonify(response), 201

#do FL with the received models
def doFL():
    print("Generating the FL model")
    # print("devices_num_epochs")
    # print(federated_learning_server.devices_num_epochs)
    # print(len(federated_learning_server.devices_hidden_layer))
    # print(len(federated_learning_server.devices_output_layer))
    # Processing models
    hidden_layer = np.average(federated_learning_server.devices_hidden_layer, axis=0, weights=federated_learning_server.devices_num_epochs)
    output_layer = np.average(federated_learning_server.devices_output_layer, axis=0, weights=federated_learning_server.devices_num_epochs)

    # Sending model to all registered devices
    # print("devices_num_epochs after clear")
    # print(federated_learning_server.devices_num_epochs)
    threads = []
    for d in federated_learning_server.client_adresses:
        thread = threading.Thread(target=FLSendModel, args=(d, hidden_layer, output_layer))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join() # Wait for all the threads to end

    print("FL has been completed sucesfully")

    # reset values for next FL
    federated_learning_server.devices_num_epochs.clear()
    federated_learning_server.devices_hidden_layer_batch_amount = np.zeros((federated_learning_server.numdevices))

    # Telling all registered devices that fl is finished
    print("Telling the devices that FL is finished")
    threads = []
    for d in federated_learning_server.client_adresses:
        print(d) 
        # have to add a coma in args if there is only 1 value, if not it will read it as a list of chars
        thread = threading.Thread(target=FLSendConfirmation, args=(d,))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join() # Wait for all the threads to end

#send the calculated model to a board
def FLSendModel(d, hidden_layer, output_layer):
    print(f"Sending FL model to device {d}")
    min_w = min(min(hidden_layer), min(output_layer))
    max_w = max(max(hidden_layer), max(output_layer))
    print(f"[{d}] Min weight to send: {min_w}, max: {max_w}")

    calculated_hidden_layer = []
    for i in range(federated_learning_server.size_hidden_layer): # hidden layer
        # if i < 5: print(f"[{d}] Sending weight {i}: {hidden_layer[i]}")
        if federated_learning_server.mixedPrecision:
            scaled = scaleWeight(min_w, max_w, hidden_layer[i])
            calculated_hidden_layer.append((scaled.to_bytes(federated_learning_server.scaledWeightsSize, "little", signed=True)))
        else:
            float_num = hidden_layer[i]
            calculated_hidden_layer.append(float_num)

    # print("calculated_hidden_layer size: ")
    # print(federated_learning_server.size_hidden_layer)
    # print(len(calculated_hidden_layer))

    calculated_output_layer = []

    for i in range(federated_learning_server.size_output_layer): # output layer
        if federated_learning_server.mixedPrecision:
            scaled = scaleWeight(min_w, max_w, output_layer[i])
            calculated_output_layer.append((scaled.to_bytes(federated_learning_server.scaledWeightsSize, "little", signed=True)))
        else:
            float_num = output_layer[i]
            calculated_output_layer.append(float_num)

    # send output layer
    print("Sending output layer")
    data_output = {
        "message" : "output layer sent correctly to board",
        "min_weight" : min_w,
        "max_weight" : max_w,
        "output_layer" : calculated_output_layer
    }
    # print("destination IP: ")
    # print(d)
    urlD_output = "http://" + d + ":80/sendOutputData"
    print(f"destination URL: {urlD_output}")
    # print("destination URL: ")
    # print(urlD_output)
    # print("data")
    # print(data)

    send_ini_time = time.time()
    with app.app_context():
        num_retries = 10
        for x in range(0, num_retries):
            try:
                r = requests.post(url = urlD_output, data = data_output, timeout= 60)
                if (r): 
                    # the message was received correctly
                    print("The message was received correctly")
                    break
                else:
                    print("ERROR. Try again")
            except WindowsError as e:
                if e.winerror == 10054:
                    print("ERROR: ConnectionResetError: [WinError 10054]. Try again")
    
    send_time = time.time()-send_ini_time
    print(f'Output layer message sent to board in ({send_time} seconds)')

    # send hidden layer
    batchSize = 1000
    batchNumber = 0
    i = 0
    print(f"Sending hidden layer in batches of {batchSize} weights")
    while i < federated_learning_server.size_hidden_layer:
        # print("All iteration: ")
        # print(i)
        print(f'Hidden layer batch  {batchNumber}')
        if (i + batchSize > federated_learning_server.size_hidden_layer):
            print("It's the last batch")
            data_hidden = {
                "message" : "hidden layer batch sent correctly to board",
                "min_weight" : min_w,
                "max_weight" : max_w,
                "batchNumber" : batchNumber,
                "last" : 1,
                "batchSize" : batchSize,
                "hidden_layer_batch" : calculated_hidden_layer[i:federated_learning_server.size_hidden_layer]
            }
            # print("destination IP: ")
            # print(d)
            urlD_hidden = "http://" + d + ":80/sendHiddenData"
            print(f"Destination URL: {urlD_hidden}")
            # print("destination URL: ")
            # print(urlD_hidden)
            # print("data")
            # print(data)

            send_ini_time = time.time()
            with app.app_context():
                num_retries = 10
                for x in range(0, num_retries):
                    try:
                        r = requests.post(url = urlD_hidden, data = data_hidden, timeout= 60)
                        if (r): 
                            # the message was received correctly
                            print("The message was received correctly")
                            break
                        else:
                            print("ERROR. Try again")
                    except WindowsError as e:
                        if e.winerror == 10054:
                            print("ERROR: ConnectionResetError: [WinError 10054]. Try again")
            
            send_time = time.time()-send_ini_time
            print(f'Hidden layer message sent to board in ({send_time} seconds)')
            
        else:
            print("It's not the last batch")
            data_hidden = {
                "message" : "hidden layer batch sent correctly to board",
                "min_weight" : min_w,
                "max_weight" : max_w,
                "batchNumber" : batchNumber,
                "last" : 0,
                "batchSize" : batchSize,
                "hidden_layer_batch" : calculated_hidden_layer[i:i + batchSize]
            }
            # print("destination IP: ")
            # print(d)
            urlD_hidden = "http://" + d + ":80/sendHiddenData"
            print(f"Destination URL: {urlD_hidden}")
            # print("destination URL: ")
            # print(urlD_hidden)
            # print("data")
            # print(data)

            send_ini_time = time.time()
            with app.app_context():
                num_retries = 10
                for x in range(0, num_retries):
                    try:
                        r = requests.post(url = urlD_hidden, data = data_hidden, timeout= 60)
                        if (r): 
                            # the message was received correctly
                            print("The message was received correctly")
                            break
                        else:
                            print("ERROR. Try again")
                    except WindowsError as e:
                        if e.winerror == 10054:
                            print("ERROR: ConnectionResetError: [WinError 10054]. Try again")
            
            send_time = time.time()-send_ini_time
            print(f'Hidden layer message sent to board in ({send_time} seconds)')
        
        batchNumber = batchNumber+1
        i = i + batchSize
    return

#send a message to tell the devices that fl is finished
def FLSendConfirmation(d):
    # print("destination IP: ")
    # print(d)
    urlD = "http://" + d + ":80/sendFLConfirmation"
    print(f"Destination URL: {urlD}")
    # print("destination URL: ")
    # print(urlD)

    send_ini_time = time.time()
    with app.app_context():
        num_retries = 10
        for x in range(0, num_retries):
            try:
                r = requests.get(url = urlD)
                if (r): 
                    # the message was received correctly
                    print("The message was received correctly")
                    break
                else:
                    print("ERROR. Try again")
            except WindowsError as e:
                if e.winerror == 10054:
                    print("ERROR: ConnectionResetError: [WinError 10054]. Try again")
    
    send_time = time.time()-send_ini_time
    print(f'FL is finished message sent to board in ({send_time} seconds)')

    return

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app.run(host='0.0.0.0', port=port)

#Get the models of all devices, merge them doing the average and send them back
# def startFL():
#     global devices_connected, hidden_layer, output_layer, pauseListen

#     pauseListen = True

#     if debug: print('Starting Federated Learning')
#     old_devices_connected = devices_connected
#     devices_connected = []
#     devices_hidden_layer = np.empty((len(devices), size_hidden_layer), dtype='float32')
#     devices_output_layer = np.empty((len(devices), size_output_layer), dtype='float32')
#     devices_num_epochs = []

#     # Receiving models
#     threads = []
#     for i, d in enumerate(devices):
#         thread = threading.Thread(target=FlGetModel, args=(d, i, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected))
#         thread.daemon = True
#         thread.start()
#         threads.append(thread)
 
#     for thread in threads: thread.join() # Wait for all the threads to end

#     print(devices_hidden_layer)

#     # Processing models

#     # if sum == 0, any device made any epoch
#     if debug: print(f"Devices num epochs: {devices_num_epochs}")
#     if sum(devices_num_epochs) > 0:
#         # We can use weights to change the importance of each device
#         # example weights = [1, 0.5] -> giving more importance to the first device...
#         # is like percentage of importance :  sum(a * weights) / sum(weights)
#         ini_time = time.time() * 1000
#         hidden_layer = np.average(devices_hidden_layer, axis=0, weights=devices_num_epochs)
#         output_layer = np.average(devices_output_layer, axis=0, weights=devices_num_epochs)
#         if debug: print(f'[{d.port}] Average millis: {(time.time()*1000)-ini_time} milliseconds)')

#     # Sending models
#     threads = []
#     for d in devices_connected:
#         if debug: print(f'[{d.port}] Sending model...')
#         thread = threading.Thread(target=sendModel, args=(d, hidden_layer, output_layer))
#         thread.daemon = True
#         thread.start()
#         threads.append(thread)
#     for thread in threads: thread.join() # Wait for all the threads to end
#     pauseListen = False