from ast import keyword
import serial
from serial.tools.list_ports import comports

import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import json
import os
import random
from queue import Queue

seed = 4321
random.seed(seed)
np.random.seed(seed)

mixedPrecision = False
scaledWeightsSize = 1
samples_per_device = 5 # Amount of samples of each word to send to each device
batch_size = 1 # Must be even, has to be split into 3 types of samples
experiment = 'iid' # 'iid', 'no-iid', 'train-test', None

debug = True

keywords_buttons = {
    "montserrat": 1,
    "pedraforca": 2,
    "vermell": 3,
    "blau": 4,
    # "verd": 5
}

output_nodes = len(keywords_buttons)
test_samples_amount = 60
size_hidden_nodes = 1 #25
size_hidden_layer = (650+1)*size_hidden_nodes #650+1
size_output_layer = (size_hidden_nodes+1)*output_nodes
momentum = 0.9
learningRate= 0.6
pauseListen = False # So there are no threads reading the serial input at the same time

montserrat_files = [file for file in os.listdir("datasets/keywords") if file.startswith("montserrat")]
pedraforca_files = [file for file in os.listdir("datasets/keywords") if file.startswith("pedraforca")]
vermell_files = [file for file in os.listdir("datasets/keywords") if file.startswith("vermell")]
blau_files = [file for file in os.listdir("datasets/keywords") if file.startswith("blau")]
verd_files = [file for file in os.listdir("datasets/keywords") if file.startswith("verd")]

test_montserrat_files = [file for file in os.listdir("datasets/test-keywords/") if file.startswith("montserrat")]
test_pedraforca_files = [file for file in os.listdir("datasets/test-keywords") if file.startswith("pedraforca")]
test_blau_files = [file for file in os.listdir("datasets/test-keywords") if file.startswith("blau")]
test_verd_files = [file for file in os.listdir("datasets/test-keywords") if file.startswith("verd")]
test_vermell_files = [file for file in os.listdir("datasets/test-keywords") if file.startswith("vermell")]

random.shuffle(montserrat_files)
random.shuffle(pedraforca_files)
random.shuffle(blau_files)
random.shuffle(verd_files)
random.shuffle(vermell_files)

keywords = list(sum(zip(montserrat_files, pedraforca_files, vermell_files, blau_files), ()))
test_keywords = list(sum(zip(test_montserrat_files, test_pedraforca_files, test_vermell_files, test_blau_files), ()))

def print_until_keyword(keyword, arduino):
    while True: 
        msg = arduino.readline().decode()
        if msg[:-2] == keyword: break
        else: print(f'({arduino.port}):',msg, end='')

def init_network(hidden_layer, output_layer, device):
    device.reset_input_buffer()
    device.write(b's')
    print_until_keyword('start', device)
    device.write(struct.pack('i', seed))
    print(f"Seed conf: {device.readline().decode()}")
    if debug: print(f"[{device.port}] Sending model to {device.port}")

    device.write(struct.pack('f', learningRate))
    device.write(struct.pack('f', momentum))

    for i in range(len(hidden_layer)):
        if i < 5 and device.port == 'com5': print(f"[{device.port}] Init Weight {i}: {hidden_layer[i]}") #tenia com6
        device.read() # wait until confirmation of float received
        device.write(struct.pack('f', hidden_layer[i]))
    
    for i in range(len(output_layer)):
        device.read() # wait until confirmation of float received
        device.write(struct.pack('f', output_layer[i]))

    if debug: print(f"Model sent to {device.port}")
    modelReceivedConfirmation = device.readline().decode()
    if debug: print(f"[{device.port}] Model received confirmation: ", modelReceivedConfirmation)
    
# Batch size: The amount of samples to send
def sendSamplesIID(device, deviceIndex, batch_size, batch_index):
    global montserrat_files, pedraforca_files, keywords

    start = ((deviceIndex*samples_per_device) + (batch_index * batch_size))#%len(keywords)
    end = ((deviceIndex*samples_per_device) + (batch_index * batch_size) + batch_size)#%len(keywords)

    if debug: print(f"[{device.port}] Sending samples from {start} to {end}")

    # files = keywords[start:end]
    for i in range(start, end):
        filename = keywords[i % len(keywords)]
        keyword = filename.split(".")[0]
        num_button = keywords_buttons[keyword]

        if debug: print(f"[{device.port}] Sending sample {filename} ({i}/{end-start}): Button {num_button}")
        error, success = sendSample(device, 'datasets/keywords/'+filename, num_button, deviceIndex)
        successes_queue_map[deviceIndex].put(success)
        errors_queue_map[deviceIndex].put(error)
        samplesAccuracyTick = sum(successes_queue_map[deviceIndex].queue)/len(errors_queue_map[deviceIndex].queue)
        print("Samples accuracy tick: {samplesAccuracyTick}")
        accuracy_map[deviceIndex].append(samplesAccuracyTick)

def sendSamplesNonIID(device, deviceIndex, batch_size, batch_index):
    global montserrat_files, pedraforca_files, blau_files, verd_files, vermell_files

    start = (batch_index * batch_size)
    end = (batch_index * batch_size) + batch_size

    if (deviceIndex == 0):
        files = vermell_files[start:end]
        num_button = 1
    elif  (deviceIndex == 1):
        files = montserrat_files[start:end]
        num_button = 2
    elif  (deviceIndex == 2):
        files = pedraforca_files[start:end]
        num_button = 3
    else:
        exit("Exceeded device index")

    for i, filename in enumerate(files):
        print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
        error, success = sendSample(device, f"datasets/keywords/{filename}", num_button, deviceIndex)
        successes_queue_map[deviceIndex].put(success)
        errors_queue_map[deviceIndex].put(error)
        accuracy_map[deviceIndex].append(sum(successes_queue_map[deviceIndex].queue)/len(errors_queue_map[deviceIndex].queue))

def sendSample(device, samplePath, num_button, deviceIndex, only_forward = False):
    with open(samplePath) as f:
        ini_time = time.time() * 1000
        data = json.load(f)
        if debug: print(f'[{device.port}] Sending train command')
        device.write(b't')
        startConfirmation = device.readline().decode()
        while (startConfirmation[:-2] != "ok"):
            if debug: print(f"[{device.port}] Incorrect confirmation:", startConfirmation)
            startConfirmation = device.readline().decode()
        if debug: print(f"[{device.port}] Train start confirmation:", startConfirmation)

        device.write(struct.pack('B', num_button))
        button_confirmation = device.readline().decode() # Button confirmation
        if debug: print(f"[{device.port}] Button confirmation: {button_confirmation}")

        device.write(struct.pack('B', 1 if only_forward else 0))
        only_forward_conf = device.readline().decode()
        if debug: print(f"[{device.port}] Only forward confirmation: {only_forward_conf}") # Button confirmation

        for i, value in enumerate(data['payload']['values']): 
            device.write(struct.pack('h', value))
            # device.read()

        sample_received_conf = device.readline().decode()
        if debug: print(f"[{device.port}] Sample received confirmation:", sample_received_conf)

        graphCommand = device.readline().decode()
        if debug: print(f"[{device.port}] Graph command received: {graphCommand}")
        error, num_button_predicted = read_graph(device, deviceIndex)
        if debug: print(f'[{device.port}] Sample sent in: {(time.time()*1000)-ini_time} milliseconds)')

    return error, num_button == num_button_predicted

def sendTestSamples(device, deviceIndex, test_accuracies_map):
    global test_keywords

    errors_queue = Queue()
    successes_queue = Queue()

    start = deviceIndex*test_samples_amount
    end = (deviceIndex*test_samples_amount) + test_samples_amount

    if debug: print(f"[{device.port}] Sending test samples from {start} to {end}")
   
    for i, filename in enumerate(test_keywords[start:end]):
        keyword = filename.split(".")[0]
        num_button = keywords_buttons[keyword]
        
        error, success = sendSample(device, 'datasets/test-keywords/'+filename, num_button, deviceIndex, True)
        errors_queue.put(error)
        successes_queue.put(success)
    
    test_accuracy = sum(successes_queue.queue)/len(successes_queue.queue)
    print(f"[{device.port}] Testing accuracy: {test_accuracy}")
    test_accuracies_map[deviceIndex] = test_accuracy

def sendTestAllDevices():
    test_accuracies_map = {}
    global devices
    for deviceIndex, device in enumerate(devices):
        thread = threading.Thread(target=sendTestSamples, args=(device, deviceIndex, test_accuracies_map))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join()
    return test_accuracies_map

def read_graph(device, deviceIndex):
    outputs = device.readline()
    print(f'[{device.port}] Outputs [raw]: {outputs}')
    outputs = outputs.decode().split()
    print(f'[{device.port}] Outputs: {outputs}')

    bpred = outputs.index(max(outputs))+1
    if debug: print(f'[{device.port}] Predicted button: {bpred}')
    
    error = device.readline().decode()
    print(f"[{device.port}] Error: {error}")

    ne = device.readline()[:-2]
    n_epooch = int(ne)

    n_error = device.read(4)
    [n_error] = struct.unpack('f', n_error)
    nb = device.readline()[:-2]
    graph.append([n_epooch, n_error, deviceIndex])
    return n_error, outputs.index(max(outputs)) + 1

def read_number(msg):
    while True:
        try:
            #return 2;
            return int(input(msg))
        except: print("ERROR: Not a number")

def read_port(msg):
    while True:
        try:
            port = input(msg)
            #port = "COM8";
            return serial.Serial(port, 9600)
        except: print(f"ERROR: Wrong port connection ({port})")


graph = []
def plot_mse_graph():
    global graph, devices

    plt.ion()
    # plt.title(f"Loss vs Epoch")
    plt.show()

    font_sm = 13
    font_md = 16
    font_xl = 18
    plt.rc('font', size=font_sm)          # controls default text sizes
    plt.rc('axes', titlesize=font_sm)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_md)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_sm)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_sm)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_sm)    # legend fontsize
    plt.rc('figure', titlesize=font_xl)   # fontsize of the figure title

    colors = ['r', 'g', 'b', 'y']
    markers = ['-', '--', ':', '-.']
    epochs = 1
    for device_index, device in enumerate(devices):
        epoch = [x[0] for x in graph if x[2] == device_index]
        error = [x[1] for x in graph if x[2] == device_index]

        epochs = max(len(error), epochs)
    
        plt.plot(error, colors[device_index] + markers[device_index], label=f"Device {device_index}")

    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=0.7)
    plt.ylabel('Loss') # or Error
    plt.xlabel('Epoch')

    # plt.axes().set_ylim([0, 0.6])
    # plt.xlim(bottom=0)
    plt.autoscale(axis='x')
    #plt.xlim = epochs
    #plt.xticks(range(0, epochs))

    if (experiment == 'train-test'): plt.axvline(x=samples_per_device)

    plt.pause(2)

    

def listenDevice(device, deviceIndex):
    global pauseListen, graph
    while True:
        while (pauseListen): time.sleep(0.1)

        device.timeout = None
        msg = device.readline().decode()
        if (len(msg) > 0):
            print(f'({device.port}):', msg, end="")
            if msg[:-2] == 'graph': read_graph(device, deviceIndex)
            #elif msg[:-2] == 'start_fl': startFL()

def getDevices():
    global devices, devices_connected
    num_devices = read_number("Number of devices: ")
    # num_devices = 2

    available_ports = comports()
    print("Available ports:")
    for available_port in available_ports: print(available_port)

    devices = [read_port(f"Port device_{i+1}: ") for i in range(num_devices)]
    devices_connected = devices

def FlGetModel(device, deviceIndex):
    print(f'[{device.port}] Starting connection...') # Handshake
    device.write(b'>') # Python --> SYN --> Arduino
    syn_ack = device.read()
    while (syn_ack[:-2] != "ok"):
        if debug: print(f"[{device.port}] Incorrect confirmation:", syn_ack)
        syn_ack = device.readline().decode()
    if debug: print(f"[{device.port}] SYN confirmation:", syn_ack)

def startFL():
    global devices
    threads = []
    for deviceIndex, device in enumerate(devices):
        print(f'[{device.port}] Starting thread...')
        thread = threading.Thread(target=FlGetModel, args=(device, deviceIndex))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join()

def readFloat(d):
    data = d.read(4)
    [float_num] = struct.unpack('f', data)
    return float_num

def readInt(d, size):
    return int.from_bytes(d.read(size), "little", signed=True)

def deScaleWeight(min_w, max_w, weight):
    a, b = getScaleRange()
    return min_w + ( (weight-a)*(max_w-min_w) / (b-a) )

def scaleWeight(min_w, max_w, weight):
    a, b = getScaleRange()
    return round(a + ( (weight-min_w)*(b-a) / (max_w-min_w) ))


def getScaleRange():
    if scaledWeightsSize == 1:
        return -128, 127
    elif scaledWeightsSize == 2:
        return -32768, 32767
    elif scaledWeightsSize == 4:
        return -2147483648, 2147483647

def sendModel(d, hidden_layer, output_layer):
    min_w = min(min(hidden_layer), min(output_layer))
    max_w = max(max(hidden_layer), max(output_layer))
    print(f"[{d.port}] Min weight to send: {min_w}, max: {max_w}")
    
    d.write(struct.pack('f', min_w))
    d.write(struct.pack('f', max_w))

    ini_time = time.time()
    for i in range(size_hidden_layer): # hidden layer
        if i < 5: print(f"[{d.port}] Sending weight {i}: {hidden_layer[i]}")
        if mixedPrecision:
            scaled = scaleWeight(min_w, max_w, hidden_layer[i])
            d.write(scaled.to_bytes(scaledWeightsSize, "little", signed=True))
        else:
            float_num = hidden_layer[i]
            d.write(struct.pack('f', float_num))

    for i in range(size_output_layer): # output layer
        if mixedPrecision:
            scaled = scaleWeight(min_w, max_w, output_layer[i])
            d.write(scaled.to_bytes(scaledWeightsSize, "little", signed=True))
        else:
            float_num = output_layer[i]
            d.write(struct.pack('f', float_num))

    if debug: print(f'Model sent to {d.port} ({time.time()-ini_time} seconds)')
    confirmation = d.readline().decode()
    if debug: print(f'Model received confirmation: {confirmation}')

# Send the blank model to all the devices
def sendInitWeights():
    threads = []
    for i, d in enumerate(devices):
        hidden_layer = np.random.uniform(-0.5,0.5, size_hidden_layer).astype('float32')
        output_layer = np.random.uniform(-0.5, 0.5, size_output_layer).astype('float32')

        thread = threading.Thread(target=init_network, args=(hidden_layer, output_layer, d))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join() # Wait for all the threads to end

# Start the execution of the script
getDevices()

sendInitWeights()

accuracy_map = {}
for deviceIndex, device in enumerate(devices): accuracy_map[deviceIndex] = []

if experiment != None:
    train_ini_time = time.time()
    errors_queue_map = {}
    successes_queue_map = {}
    for deviceIndex, device in enumerate(devices):
        errors_queue_map[deviceIndex] = Queue() # MSE errors
        successes_queue_map[deviceIndex] = Queue() # Amount of right inferences

    threads = []
    # Train the device
    for batch in range(int(samples_per_device/batch_size)):
        batch_ini_time = time.time()
        for deviceIndex, device in enumerate(devices):
            if experiment == 'iid' or experiment == 'train-test': method = sendSamplesIID
            elif experiment == 'no-iid': method = sendSamplesNonIID

            thread = threading.Thread(target=method, args=(device, deviceIndex, batch_size, batch))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            
        for thread in threads: thread.join() # Wait for all the threads to end
        print(f'Batch time: {time.time() - batch_ini_time} seconds)')
        # fl_ini_time = time.time()
        # startFL()
        # print(f'FL time: {time.time() - fl_ini_time} seconds)')
        # time.sleep(1)

    train_time = time.time()-train_ini_time
    if debug: print(f'Trained in ({train_time} seconds)')

    for deviceIndex, device in enumerate(devices):
        print(f"[{device.port}] Training loss total: {sum(errors_queue_map[deviceIndex].queue)}")
        print(f"[{device.port}] Training loss mean: {sum(errors_queue_map[deviceIndex].queue)/len(errors_queue_map[deviceIndex].queue)}")
        print(f"[{device.port}] Training accuracy: {sum(successes_queue_map[deviceIndex].queue)/len(successes_queue_map[deviceIndex].queue)}")


if experiment == 'train-test': 
    test_accuracies_map = sendTestAllDevices()
    print(test_accuracies_map)



# plot_mse_graph()

plt.ylim(bottom=0, top=1)
plt.xlim(left=0)
plt.autoscale(axis='x')
for device_index, device in enumerate(devices):
   plt.plot(accuracy_map[device_index], label=f"Device {device_index}")

if experiment != None:
    # figname = f"plots/BS{batch_size}-LR{learningRate}-M{momentum}-HL{size_hidden_nodes}-TT{train_time}-{experiment}.png"
    figname = f"plots/{len(devices)}-{scaledWeightsSize if mixedPrecision else 'no'}.png"
    plt.savefig(figname, format='png')
    print(f"Generated {figname}")

# plt.show()