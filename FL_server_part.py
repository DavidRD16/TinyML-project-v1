import hashlib
import json
import threading
from time import time
from urllib.parse import urlparse
from uuid import uuid4
import numpy as np

import requests
from flask import Flask, jsonify, request


class Federated_learning_server:
    def __init__(self):
        #blockchain
        self.current_transactions = []
        self.chain = []
        self.nodes = set()

        # Create the genesis block
        self.new_block(previous_hash='1', proof=100)

        #federated learning
        self.client_adresses = []
        
        self.mixedPrecision = True
        self.scaledWeightsSize = 1
        keywords_buttons = {
            "montserrat": 1,
            "pedraforca": 2,
            "vermell": 3,
            "blau": 4,
            # "verd": 5
        }
        numdevices = 1
        input_nodes = 65
        output_nodes = len(keywords_buttons)
        size_hidden_nodes = 25
        self.size_hidden_layer = (input_nodes+1)*size_hidden_nodes
        self.size_output_layer = (size_hidden_nodes+1)*output_nodes
        self.devices_hidden_layer = np.empty((numdevices, self.size_hidden_layer), dtype='float32')
        self.devices_output_layer = np.empty((numdevices, self.size_output_layer), dtype='float32')

        self.devices_num_epochs = []

    def register_node(self, address):
        """
        Add a new node to the list of nodes

        :param address: Address of node. Eg. 'http://192.168.0.5:5000'
        """

        parsed_url = urlparse(address)
        if parsed_url.netloc:
            self.nodes.add(parsed_url.netloc)
        elif parsed_url.path:
            # Accepts an URL without scheme like '192.168.0.5:5000'.
            self.nodes.add(parsed_url.path)
        else:
            raise ValueError('Invalid URL')


    def valid_chain(self, chain):
        """
        Determine if a given blockchain is valid

        :param chain: A blockchain
        :return: True if valid, False if not
        """

        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            print(f'{last_block}')
            print(f'{block}')
            print("\n-----------\n")
            # Check that the hash of the block is correct
            if block['previous_hash'] != self.hash(last_block):
                return False

            # Check that the Proof of Work is correct
            #if not self.valid_proof(last_block['proof'], block['proof'], last_block['previous_hash']):
            if not self.valid_proof(last_block['proof'], block['proof'], block['previous_hash']):
                return False

            last_block = block
            current_index += 1

        return True

    def resolve_conflicts(self):
        """
        This is our consensus algorithm, it resolves conflicts
        by replacing our chain with the longest one in the network.

        :return: True if our chain was replaced, False if not
        """

        neighbours = self.nodes
        new_chain = None

        # We're only looking for chains longer than ours
        max_length = len(self.chain)

        # Grab and verify the chains from all the nodes in our network
        for node in neighbours:
            response = requests.get(f'http://{node}/chain')

            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']

                # Check if the length is longer and the chain is valid
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain

        # Replace our chain if we discovered a new, valid chain longer than ours
        if new_chain:
            self.chain = new_chain
            return True

        return False

    def new_block(self, proof, previous_hash):
        """
        Create a new Block in the Blockchain

        :param proof: The proof given by the Proof of Work algorithm
        :param previous_hash: Hash of previous Block
        :return: New Block
        """

        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }

        # Reset the current list of transactions
        self.current_transactions = []

        self.chain.append(block)
        return block

    def new_transaction(self, sender, recipient, amount, order):
        """
        Creates a new transaction to go into the next mined Block

        :param sender: Address of the Sender
        :param recipient: Address of the Recipient
        :param amount: Amount
        :param order: Order number
        :return: The index of the Block that will hold this transaction
        """
        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
            'order': order,
        })

        return self.last_block['index'] + 1

    @property
    def last_block(self):
        return self.chain[-1]

    @staticmethod
    def hash(block):
        """
        Creates a SHA-256 hash of a Block

        :param block: Block
        """

        # We must make sure that the Dictionary is Ordered, or we'll have inconsistent hashes
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def proof_of_work(self, last_block):
        """
        Simple Proof of Work Algorithm:

         - Find a number p' such that hash(pp') contains leading 4 zeroes
         - Where p is the previous proof, and p' is the new proof

        :param last_block: <dict> last Block
        :return: <int>
        """

        last_proof = last_block['proof']
        last_hash = self.hash(last_block)

        proof = 0
        while self.valid_proof(last_proof, proof, last_hash) is False:
            proof += 1

        return proof

    @staticmethod
    def valid_proof(last_proof, proof, last_hash):
        """
        Validates the Proof

        :param last_proof: <int> Previous Proof
        :param proof: <int> Current Proof
        :param last_hash: <str> The hash of the Previous Block
        :return: <bool> True if correct, False if not.

        """

        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"


# Instantiate the Node
app = Flask(__name__)

# Generate a globally unique address for this node
node_identifier = str(uuid4()).replace('-', '')

# Instantiate the server
federated_learning_server = Federated_learning_server()


@app.route('/mine', methods=['GET'])
def mine():
    # We run the proof of work algorithm to get the next proof...
    last_block = federated_learning_server.last_block
    proof = federated_learning_server.proof_of_work(last_block)

    # We must receive a reward for finding the proof.
    # The sender is "0" to signify that this node has mined a new coin.
    federated_learning_server.new_transaction(
        sender="0",
        recipient=node_identifier,
        amount=1,
        order=0,
    )

    # Forge the new Block by adding it to the chain
    previous_hash = federated_learning_server.hash(last_block)
    block = federated_learning_server.new_block(proof, previous_hash)

    response = {
        'message': "New Block Forged",
        'index': block['index'],
        'transactions': block['transactions'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
    }
    return jsonify(response), 200


@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.get_json()

    # Check that the required fields are in the POST'ed data
    required = ['sender', 'recipient', 'amount', 'order']
    if not all(k in values for k in required):
        return 'Missing values', 400

    # Create a new Transaction
    index = federated_learning_server.new_transaction(values['sender'], values['recipient'], values['amount'], values['order'])

    response = {'message': f'Transaction will be added to Block {index}'}
    return jsonify(response), 201


@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': federated_learning_server.chain,
        'length': len(federated_learning_server.chain),
    }
    return jsonify(response), 200


@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.get_json()

    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Please supply a valid list of nodes", 400

    #for node in nodes:
    #    blockchain.register_node(node)
    federated_learning_server.register_node(nodes)

    response = {
        'message': 'New nodes have been added',
        'total_nodes': list(federated_learning_server.nodes),
    }
    return jsonify(response), 201


@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = federated_learning_server.resolve_conflicts()

    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': federated_learning_server.chain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': federated_learning_server.chain
        }

    return jsonify(response), 200

#functions for my Portenta Project
@app.route('/test/post', methods=['POST'])
def test_post():
    values = request.get_json()
    
    test = values.get('message')
    response = {
        'message': 'POST message received',
        'content': test,
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
def FlGetModel():
    #check if the board has contacted before, and if not add its address to the list
    adress = request.remote_addr
    if not (adress in federated_learning_server.client_adresses):
        #the adress isn't in the list, so we add it
        federated_learning_server.client_adresses.append(adress)
    
    #get device index from the list of adresses
    device_index = federated_learning_server.client_adresses.index(request.remote_addr)
    values = request.get_json()
    
    num_epochs = values.get('num_epochs') #int
    print("num_epochs: ")
    print(num_epochs)
    federated_learning_server.devices_num_epochs.append(num_epochs)

    min_w = values.get('min_weight') #float
    max_w = values.get('max_weight') #float
 
    print("min_weight: ")
    print(min_w)
    print("max_weight: ")
    print(max_w)
    a, b = getScaleRange()

    #  Receiving model...
    print("HiddenWeights: ")
    received_hidden_layer = values.get('HiddenWeights')
    print(received_hidden_layer)
    for i in range(federated_learning_server.size_hidden_layer): # hidden layer
        if federated_learning_server.mixedPrecision:
            scaledWeight = received_hidden_layer[i] #int
            #scaled_in_float = readFloat(d)
            # float_weight = readFloat(d)
            weight = deScaleWeight(min_w, max_w, scaledWeight)
            # if i < 5 and d.port == 'com6': print(f"[{d.port}] Recevied Weight {i}: {float_weight}")
            # if abs(float_weight - weight) > 0.3: print(f"[{d.port}] Scaled weight: {scaledWeight} (float: {scaled_in_float}), Float weight (hid): {float_weight}, descaled: {weight}. Difference: {abs(float_weight - weight)}")
        else: weight = received_hidden_layer[i] #float
        
        # Received Weight {i}: {weight}")
        federated_learning_server.devices_hidden_layer[device_index][i] = weight

    
    print("OutputWeights: ")
    received_output_layer = values.get('OutputWeights')
    print(received_output_layer)
    for i in range(federated_learning_server.size_output_layer): # output layer
        if federated_learning_server.mixedPrecision:
            scaledWeight = received_output_layer[i] #int
            #scaled_in_float = readFloat(d)
            # float_weight = readFloat(d)
            weight = deScaleWeight(min_w, max_w, scaledWeight)
            #if abs(float_weight - weight) > 0.3: print(f"[{d.port}] Scaled weight: {scaledWeight} (float: {scaled_in_float}), Float weight (hid): {float_weight}, descaled: {weight}. Difference: {abs(float_weight - weight)}")
        else: weight = received_output_layer[i] #float
        
        # Received Weight {i}: {weight}")
        federated_learning_server.devices_output_layer[device_index][i] = weight

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
    # Processing models
    hidden_layer = np.average(federated_learning_server.devices_hidden_layer, axis=0, weights=federated_learning_server.devices_num_epochs)
    output_layer = np.average(federated_learning_server.devices_output_layer, axis=0, weights=federated_learning_server.devices_num_epochs)

    # Sending model to all registered devices
    threads = []
    for d in federated_learning_server.client_adresses:
        thread = threading.Thread(target=FLSendModel, args=(d, hidden_layer, output_layer))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads: thread.join() # Wait for all the threads to end

#send the calculated model to a board
def FLSendModel(d, hidden_layer, output_layer):
    min_w = min(min(hidden_layer), min(output_layer))
    max_w = max(max(hidden_layer), max(output_layer))
    print(f"[{d.port}] Min weight to send: {min_w}, max: {max_w}")

    calculated_hidden_layer = [federated_learning_server.size_hidden_layer]
    for i in range(federated_learning_server.size_hidden_layer): # hidden layer
        if i < 5: print(f"[{d.port}] Sending weight {i}: {hidden_layer[i]}")
        if federated_learning_server.mixedPrecision:
            scaled = scaleWeight(min_w, max_w, hidden_layer[i])
            calculated_hidden_layer[i] = (scaled.to_bytes(federated_learning_server.scaledWeightsSize, "little", signed=True))
        else:
            float_num = hidden_layer[i]
            calculated_hidden_layer[i] = float_num
    
    calculated_output_layer = [federated_learning_server.size_output_layer]
    for i in range(federated_learning_server.size_output_layer): # output layer
        if federated_learning_server.mixedPrecision:
            scaled = scaleWeight(min_w, max_w, output_layer[i])
            calculated_output_layer[i] = (scaled.to_bytes(federated_learning_server.scaledWeightsSize, "little", signed=True))
        else:
            float_num = output_layer[i]
            calculated_output_layer[i] = float_num

    data = {
        "message" : "FL model sent correclty to board",
        # "min_weight" : min_w,
        # "max_weight" : max_w,
        # "hidden_layer" : calculated_hidden_layer,
        # "output_layer" : calculated_output_layer
    }
    requests.post(url = d, json = jsonify(data))
    

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=80, type=int, help='port to listen on')
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

