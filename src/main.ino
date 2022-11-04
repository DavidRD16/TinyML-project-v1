// This program uses several functionalities and modifications 
// from the EdgeImpulse inferencing library.

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0


/* Includes ---------------------------------------------------------------- */
#include <training_kws_inference.h>
#include "neural_network.h"
#include <SPI.h>
#include <WiFi.h>
#include <Arduino_JSON.h>

char ssid[] = "yourSSID";      //  your network SSID (name)
char pass[] = "yourPASS";   // your network password
int keyIndex = 0;            // your network key Index number (needed only for WEP)

int status = WL_IDLE_STATUS;

// Initialize the Wifi client library

WiFiClient client;
WiFiServer receivingServer(80);

// server address: IP address of the flask server
IPAddress server(999,999,999,999);
String stringHost = "999.999.999.999";

/** Audio buffers, pointers and selectors */
typedef struct {
    int16_t buffer[EI_CLASSIFIER_RAW_SAMPLE_COUNT];
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal


//uint8_t num_button = 0; // 0 represents none
bool button_pressed = false;

// Defaults: 0.3, 0.9
static NeuralNetwork myNetwork;
const float threshold = 0.6;

uint16_t num_epochs = 0;

bool mixed_precision = true;
typedef int8_t scaledType;

/**
 * @brief      Arduino setup function
 */
void setup() {
    Serial.begin(9600);
    Serial.setTimeout(5000); // Default 1000

    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
    
    connect_to_wifi();
    //delay(10000);
    //register_to_FLserver(); no funciona
    init_network_model();
}

void connect_to_wifi() {
  // check for the presence of the shield:
  if (WiFi.status() == WL_NO_SHIELD) {
    Serial.println("WiFi shield not present");
    // don't continue:
    while (true);
  }

  String fv = WiFi.firmwareVersion();

  if (fv != "1.1.0") {
    Serial.println("Please upgrade the firmware");
  }

  // attempt to connect to Wifi network:

  while (status != WL_CONNECTED) {

    Serial.print("Attempting to connect to SSID: ");
    Serial.println(ssid);

    // Connect to WPA/WPA2 network. Change this line if using open or WEP network:
    status = WiFi.begin(ssid, pass);

    // wait 10 seconds for connection:
    delay(10000);
  }

  // you're connected now, so print out the status:
  printWifiStatus();
}

void register_to_FLserver(){
    // close any connection before send a new request.
  // This will free the socket on the WiFi shield
  client.stop();

  Serial.println("registering...");
  // if there's a successful connection:
  if (client.connect(server, 80)) {
      Serial.println("connected to register");
      // send the HTTP PUT request:
    client.println("GET /register HTTP/1.1");
    client.println("Host: " + stringHost);
    client.println("User-Agent: ArduinoWiFi/1.1");
    client.println("Connection: close");
  }
  else{
    // if you couldn't make a connection:
    Serial.println("register failed");
  }

}

void init_network_model() {
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDG, LOW);
    char startChar;
    do {
        startChar = Serial.read();
        //Serial.println("Waiting for new model...");
    } while(startChar != 's'); // s -> START

    Serial.println("start");
    int seed = readInt();
    srand(seed);
    Serial.println("Seed: " + String(seed));
    float learningRate = readFloat();
    float momentum = readFloat();

    myNetwork.initialize(learningRate, momentum);

    char* myHiddenWeights = (char*) myNetwork.get_HiddenWeights();
    for (uint16_t i = 0; i < (InputNodes+1) * HiddenNodes; ++i) {
        Serial.write('n');
        while(Serial.available() < 4) {}
        for (int n = 0; n < 4; n++) {
            myHiddenWeights[i*4] = Serial.read();
        }
    }

    char* myOutputWeights = (char*) myNetwork.get_OutputWeights();
    for (uint16_t i = 0; i < (HiddenNodes+1) * OutputNodes; ++i) {
        Serial.write('n');
        while(Serial.available() < 4) {}
        for (int n = 0; n < 4; n++) {
            myOutputWeights[i*4+n] = Serial.read();
        }
    }

    Serial.println("Received new model.");
}

float readFloat() {
    byte res[4];
    while(Serial.available() < 4) {}
    for (int n = 0; n < 4; n++) {
        res[n] = Serial.read();
    }
    return *(float *)&res;
}

int readInt() {
    byte res[4];
    while(Serial.available() < 4) {}
    for (int n = 0; n < 4; n++) {
        res[n] = Serial.read();
    }
    
    return *(int *)&res;
}

void train(int nb, bool only_forward) {
    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;
    ei::matrix_t features_matrix(1, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);

    EI_IMPULSE_ERROR r = get_one_second_features(&signal, &features_matrix, debug_nn);
    if (r != EI_IMPULSE_OK) {
        Serial.println("ERR: Failed to get features ("+String(r));
        return;
    }

    float myTarget[OutputNodes] = {0};
    myTarget[nb-1] = 1.f; // button 1 -> {1,0,0};  button 2 -> {0,1,0};  button 3 -> {0,0,1}

    // FORWARD
    float forward_error = myNetwork.forward(features_matrix.buffer, myTarget);
    
    // BACKWARD
    if (!only_forward) {
        myNetwork.backward(features_matrix.buffer, myTarget);
        ++num_epochs;
    }

    // Info to plot
    Serial.println("graph");

    // Print outputs as a single line
    float* output = myNetwork.get_output();
    for (size_t i = 0; i < OutputNodes; i++) {
        ei_printf_float(output[i]);
        Serial.print(" ");
    }
    Serial.print("\n");

    // Print error
    ei_printf_float(forward_error);
    Serial.print("\n");

    Serial.println(num_epochs, DEC);

    char* myError = (char*) &forward_error;
    Serial.write(myError, sizeof(float));
    
    Serial.println(nb, DEC);
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop() {
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);

    if (Serial.available()) {
        char read = Serial.read();
        if (read == '>') {
            startFL();
        } else if (read == 't') {
            receiveSampleAndTrain();
            sendDataFL();
            // receiveDataFL();
        } else { // Error
            Serial.println("Unknown command " + read);
            while(true){
                digitalWrite(LEDR, LOW);
                delay(100);
                digitalWrite(LEDR, HIGH);
            }
        }
    }

    delay(50);
    digitalWrite(LEDG, LOW);
    delay(50);
}

void receiveSampleAndTrain() {
    digitalWrite(LEDR, LOW);
    
    Serial.println("ok");

    while(Serial.available() < 1) {}
    uint8_t num_button = Serial.read(); //num_button == categoria de la muestra, ex: Montserrat=0 y Pedraforca=1
    Serial.println("Button " + String(num_button));

    while(Serial.available() < 1) {}
    bool only_forward = Serial.read() == 1;
    Serial.println("Only forward " + String(only_forward));
    
    //recibir data del fichero en samplePath
    byte ref[2];
    for(int i = 0; i < EI_CLASSIFIER_RAW_SAMPLE_COUNT; i++) {
        while(Serial.available() < 2) {}
        Serial.readBytes(ref, 2);
        inference.buffer[i] = 0;
        inference.buffer[i] = (ref[1] << 8) | ref[0];
        // Serial.write(1);
    }
    Serial.println("Sample received for button " + String(num_button));
    train(num_button, only_forward);
}

void sendDataFL(){
// close any connection before send a new request.
  // This will free the socket on the WiFi shield
  client.stop();
  Serial.println("sending data...");
  // if there's a successful connection:
  if (client.connect(server, 80)) {
    Serial.println("connecting...");

    //send HTTP POST request
    client.println("POST /FL/sendData HTTP/1.1");
    client.println("Host: " + stringHost);
    client.println("User-Agent: ArduinoWiFi/1.1");
    client.println("Connection: close");
    client.println(F("Content-Type: application/json"));
    JSONVar myJSONObject;
    myJSONObject["message"] = ">";
    myJSONObject["num_epochs"] = num_epochs;

    Serial.println("computing min and max weights...");
    // Find min and max weights
        float* float_hidden_weights = myNetwork.get_HiddenWeights();
        float* float_output_weights = myNetwork.get_OutputWeights();
        float min_weight = float_hidden_weights[0];
        float max_weight = float_hidden_weights[0];
        for(uint i = 0; i < hiddenWeightsAmt; i++) {
            if (min_weight > float_hidden_weights[i]) min_weight = float_hidden_weights[i];
            if (max_weight < float_hidden_weights[i]) max_weight = float_hidden_weights[i];
        }
        for(uint i = 0; i < outputWeightsAmt; i++) {
            if (min_weight > float_output_weights[i]) min_weight = float_output_weights[i];
            if (max_weight < float_output_weights[i]) max_weight = float_output_weights[i];
        }

    myJSONObject["min_weight"] = (float) min_weight;
    myJSONObject["max_weight"] = (float) max_weight;
        // Serial.write((byte *) &min_weight, sizeof(float));
        // Serial.write((byte *) &max_weight, sizeof(float));

    Serial.println("computing HiddenWeights...");
    // Serial.print("the number of iterations is: ");
    // Serial.println(hiddenWeightsAmt);
    JSONVar castedHiddenWeights;
        // Sending hidden layer
        char* hidden_weights = (char*) myNetwork.get_HiddenWeights();
        for (uint16_t i = 0; i < hiddenWeightsAmt; ++i) {
            // Serial.print("iteration: ");
            // Serial.println(i);
            if (mixed_precision) {
                scaledType weight = scaleWeight(min_weight, max_weight, float_hidden_weights[i]);
                scaledType casted = weight;
                //Serial.write((byte*) &casted, sizeof(scaledType));
                castedHiddenWeights[i] = casted;
            } else {
                //Serial.write((byte*) &float_hidden_weights[i], sizeof(float)); // debug
                castedHiddenWeights[i] = float_hidden_weights[i];
            }
        }
    
    myJSONObject["HiddenWeights"] = castedHiddenWeights;

    Serial.println("computing OutputWeights...");
    JSONVar castedOutputWeights;
        // Sending output layer
        char* output_weights = (char*) myNetwork.get_OutputWeights();
        for (uint16_t i = 0; i < outputWeightsAmt; ++i) {
            if (mixed_precision) {
                scaledType weight = scaleWeight(min_weight, max_weight, float_output_weights[i]);
                scaledType casted = weight;
                //Serial.write((byte*) &casted, sizeof(scaledType));
                castedOutputWeights[i] = casted;
            } else {
                //Serial.write((byte*) &float_output_weights[i], sizeof(float)); // debug
                castedOutputWeights[i] = float_output_weights[i];
            }
        }
        
    myJSONObject["OutputWeights"] = castedOutputWeights;

        // while(!Serial.available()) {
        //     digitalWrite(LEDB, HIGH);
        //     delay(100);
        //     digitalWrite(LEDB, LOW);
        // }

    String jsonString = JSON.stringify(myJSONObject);
    Serial.print("JSON.stringify(myJSONObject) = ");
    Serial.println(jsonString);
    
    client.print(F("Content-Length: "));
    client.println(jsonString.length());
    client.println(); //hay que dejar una linea vacía
    client.println(jsonString);
    Serial.println(F("Data were sent successfully"));

    client.println();

  } else {
    // if you couldn't make a connection:
    Serial.println("connection failed");
  }
}

//wait for data from the FLserver
void receiveDataFL(){
    Serial.println("receiving data");
    String receivedData;
    bool currentLineIsBlank = true;
    while (client.connected()) {
        Serial.println("messagge received");
        char c = client.read();
        Serial.write(c);
        receivedData += c;
        // if you've gotten to the end of the line (received a newline
        // character) and the line is blank, the http request has ended,
        // so you can send a reply

        if (c == '\n' && currentLineIsBlank) {
            Serial.println();
            Serial.println(receivedData);
          // send a standard http response header
        }
        if (c == '\n') {
          // you're starting a new line
          currentLineIsBlank = true;
        } else if (c != '\r') {
          // you've gotten a character on the current line
          currentLineIsBlank = false;
        }
    }
}

void startFL() {
    digitalWrite(LEDB, LOW);
    Serial.write('<');
    while(!Serial.available()) {}
    if (Serial.read() == 's') {
        Serial.println("start");
        Serial.println(num_epochs);
        num_epochs = 0;

        // Find min and max weights
        float* float_hidden_weights = myNetwork.get_HiddenWeights();
        float* float_output_weights = myNetwork.get_OutputWeights();
        float min_weight = float_hidden_weights[0];
        float max_weight = float_hidden_weights[0];
        for(uint i = 0; i < hiddenWeightsAmt; i++) {
            if (min_weight > float_hidden_weights[i]) min_weight = float_hidden_weights[i];
            if (max_weight < float_hidden_weights[i]) max_weight = float_hidden_weights[i];
        }
        for(uint i = 0; i < outputWeightsAmt; i++) {
            if (min_weight > float_output_weights[i]) min_weight = float_output_weights[i];
            if (max_weight < float_output_weights[i]) max_weight = float_output_weights[i];
        }

        Serial.write((byte *) &min_weight, sizeof(float));
        Serial.write((byte *) &max_weight, sizeof(float));
        // Serial.write(sizeof(scaledType));

        // Sending hidden layer
        char* hidden_weights = (char*) myNetwork.get_HiddenWeights();
        for (uint16_t i = 0; i < hiddenWeightsAmt; ++i) {
            if (mixed_precision) {
                scaledType weight = scaleWeight(min_weight, max_weight, float_hidden_weights[i]);
                scaledType casted = weight;
                Serial.write((byte*) &casted, sizeof(scaledType));
            } else {
                Serial.write((byte*) &float_hidden_weights[i], sizeof(float)); // debug
            }
        }

        // Sending output layer
        char* output_weights = (char*) myNetwork.get_OutputWeights();
        for (uint16_t i = 0; i < outputWeightsAmt; ++i) {
            if (mixed_precision) {
                scaledType weight = scaleWeight(min_weight, max_weight, float_output_weights[i]);
                scaledType casted = weight;
                Serial.write((byte*) &casted, sizeof(scaledType));
            } else {
                Serial.write((byte*) &float_output_weights[i], sizeof(float)); // debug
            }
        }

        while(!Serial.available()) {
            digitalWrite(LEDB, HIGH);
            delay(100);
            digitalWrite(LEDB, LOW);
        }

        float min_received_w = readFloat();
        float max_received_w = readFloat();

        // Receiving hidden layer
        for (uint16_t i = 0; i < hiddenWeightsAmt; ++i) {
            if (mixed_precision) {
                scaledType val;
                Serial.readBytes((byte*) &val, sizeof(scaledType));
                float_hidden_weights[i] = deScaleWeight(min_received_w, max_received_w, val);
            } else {
                while(Serial.available() < 4) {}
                for (int n = 0; n < 4; n++) {
                    hidden_weights[i*4+n] = Serial.read();
                }
            }
        }
        // Receiving output layer
        for (uint16_t i = 0; i < outputWeightsAmt; ++i) {
            if (mixed_precision) {
                scaledType val;
                Serial.readBytes((byte*) &val, sizeof(scaledType));
                float_output_weights[i] = deScaleWeight(min_received_w, max_received_w, val);
            } else {
                while(Serial.available() < 4) {}
                for (int n = 0; n < 4; n++) {
                    output_weights[i*4+n] = Serial.read();
                }
            }
        }
        Serial.println("Model received");
    }
}

float scaleWeight(float min_w, float max_w, float weight) {
    float a, b;
    getScaleRange(a, b);
    return round(a + ( (weight-min_w)*(b-a) / (max_w-min_w) ));
}

float deScaleWeight(float min_w, float max_w, scaledType weight) {
    float a, b;
    getScaleRange(a, b);
    return min_w + ( (weight-a)*(max_w-min_w) / (b-a) );
}

void getScaleRange(float &a, float &b) {
    int scaledWeightSize = sizeof(scaledType);
    if (scaledWeightSize == 1) {
        a = -128;
        b = 127;
    }
    else if (scaledWeightSize == 2) {
        a = -32768;
        b = 32767;
    }
    else if (scaledWeightSize == 4) {
        a = -2147483648;
        b = 2147483647;
    }
} 


static scaledType microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
    return 0;
}

void printWifiStatus() {
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your WiFi shield's IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print the received signal strength:
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
}