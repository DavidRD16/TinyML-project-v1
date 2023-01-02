// This program uses several functionalities and modifications 
// from the EdgeImpulse inferencing library.

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0


/* Includes ---------------------------------------------------------------- */
#include <training_kws_inference.h>
#include "neural_network.h"
#include <SPI.h>
#include <WiFi.h>
#include <ArduinoJson.h>
#include <Portenta_H7_AsyncWebServer.h>
#include <StreamUtils.h>

//set to true to see prints during the execution to see the current operation
boolean debug_text = false;

//number of the device in all of the connected devices
int deviceIndex;

char ssid[] = "yourSSID";      //  your network SSID (name)
char pass[] = "yourPASS";   // your network password

int status = WL_IDLE_STATUS;

// Initialize the Wifi client library1

WiFiClient client;
WiFiServer receivingServer(5000);

// server address: IP address of the flask server
IPAddress server(999,999,999,999);
String stringHost = "999.999.999.999";

// define chunk of bytes to send when writing messages in wifi
int byteCapacity = 1024;
WriteBufferingClient bufferedWifiClient{client, byteCapacity};

bool waitingForFL = false;

// time to wait to receive a response for any message sent with wifi
int responseDelay = 5000;

unsigned long FLTime = 0;

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

bool mixed_precision = false;
typedef int8_t scaledType;

//define all functions in case of using .cpp. Not necessary when using .ino
void handleRoot(AsyncWebServerRequest *request);
void handleReceiveMainData(AsyncWebServerRequest *request);
void handleReceiveOutputData(AsyncWebServerRequest *request);
void handleReceiveHiddenData(AsyncWebServerRequest *request);
void handleReceiveFLConfirmation(AsyncWebServerRequest *request);
void handleNotFound(AsyncWebServerRequest *request);
void initAsyncServer();
void setup();
void connect_to_wifi();
void register_to_FLserver();
void init_network_model();
float readFloat();
int readInt();
void train(int nb, bool only_forward);
void loop();
void receiveSampleAndTrain();
void sendDataFL();
void sendOutputNodes();
void sendHiddenNode(uint16_t batchNumber, uint16_t lastInBatch, uint16_t batchSize,
                    uint16_t start, uint16_t end, uint16_t numBatches);
void sendAllHiddenNodes();
void receiveDataFL();
float scaleWeight(float min_w, float max_w, float weight);
float deScaleWeight(float min_w, float max_w, scaledType weight);
void getScaleRange(float &a, float &b);
static scaledType microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr);
void printWifiStatus();

//AsyncWebServer stuff
AsyncWebServer    AsyncServer(80); //80 to match FL_server
void handleRoot(AsyncWebServerRequest *request)
{
    char temp[64];
	snprintf(temp, 64 - 1, "Hello from Async_HelloServer on %s\n", BOARD_NAME);

	request->send(200, "text/plain", temp);
}

//if all data is sent in a single message (only works with small networks)
void handleReceiveMainData(AsyncWebServerRequest *request)
{
	if (request->method() != HTTP_POST)
	{
		request->send(405, "text/plain", "Method Not Allowed");
	}
	else
	{
        if (debug_text){
            Serial.println(F("AsyncWebServer msg received"));
        }

        //List all parameters (Compatibility)
        int args = request->args();
        // Serial.print("ARG[min_weight]: ");
        // Serial.println(request->arg("min_weight").c_str());
        // Serial.print("ARG[max_weight]: ");
        // Serial.println(request->arg("max_weight").c_str());
        // Serial.print("ARG[output_layer]: ");
        // Serial.println(request->arg("output_layer").c_str());

        //obtain the data from the FL server message
        float* float_hidden_weights = myNetwork.get_HiddenWeights();
        float* float_output_weights = myNetwork.get_OutputWeights();
        float min_received_w;
        float max_received_w;

        int counterHL = 0;
        int counterOL = 0;
        for (int i=0;i<args;i++)
        {
            // Serial.print("ARG[");
            // Serial.print(request->argName(i).c_str());
            // Serial.print("]: ");
            // Serial.println(request->arg(i).c_str());
            if(request->argName(i) == "min_weight"){
                min_received_w = request->arg(i).toFloat();
            }
            else if(request->argName(i) == "min_weight"){
                max_received_w = request->arg(i).toFloat();
            }
            else if(request->argName(i) == "hidden_layer"){
                // Serial.print(F("original value HL = "));
                // Serial.println(float_hidden_weights[counterHL],9);
                if (mixed_precision) {
                    scaledType val;
                    val = int(request->arg(i).c_str());
                    float_hidden_weights[counterHL] = deScaleWeight(min_received_w, max_received_w, val);
                } else {
                    float_hidden_weights[counterHL] = request->arg(i).toFloat();
                }
                // Serial.print(F("received value HL = "));
                // Serial.println(request->arg(i).toFloat(),9);
                // Serial.print(F("new value HL = "));
                // Serial.println(float_hidden_weights[counterHL],9);
                counterHL++;
            }
            else if(request->argName(i) == "output_layer"){
                // Serial.print(F("original value OL = "));
                // Serial.println(float_output_weights[counterOL],9);
                if (mixed_precision) {
                    scaledType val;
                    val = int(request->arg(i).c_str());
                    float_output_weights[counterOL] = deScaleWeight(min_received_w, max_received_w, val);
                } else {
                    float_output_weights[counterOL] = request->arg(i).toFloat();
                }
                // Serial.print(F("received value OL = "));
                // Serial.println(request->arg(i).toFloat(),9);
                // Serial.print(F("new value OL = "));
                // Serial.println(float_output_weights[counterOL],9);
                counterOL++;
            }
        }
        // for (int i=0;i<args;i++)
        // {
        //     Serial.print("ARG[");
        //     Serial.print(request->argName(i).c_str());
        //     Serial.println("]: ");
        //     Serial.println(request->arg(i).c_str());
        // }

        waitingForFL = false;
        Serial.print(F("waitingForFL = "));
        Serial.println(waitingForFL);
        request->send(200, "text/plain", "POST");
    }
}

void handleReceiveOutputData(AsyncWebServerRequest *request)
{
	if (request->method() != HTTP_POST)
	{
		request->send(405, "text/plain", "Method Not Allowed");
	}
	else
	{
        if (debug_text){
            Serial.println(F("AsyncWebServer output layer msg received"));
        }

        //List all parameters (Compatibility)
        int args = request->args();
        // Serial.print("ARG[min_weight]: ");
        // Serial.println(request->arg("min_weight").c_str());
        // Serial.print("ARG[max_weight]: ");
        // Serial.println(request->arg("max_weight").c_str());
        // Serial.print("ARG[output_layer]: ");
        // Serial.println(request->arg("output_layer").c_str());

        //obtain the data from the FL server message
        float* float_output_weights = myNetwork.get_OutputWeights();
        float min_received_w;
        float max_received_w;

        int counterOL = 0;
        for (int i=0;i<args;i++)
        {
            // Serial.print("ARG[");
            // Serial.print(request->argName(i).c_str());
            // Serial.print("]: ");
            // Serial.println(request->arg(i).c_str());
            if(request->argName(i) == "output_layer"){
                // Serial.print(F("original value OL = "));
                // Serial.println(float_output_weights[counterOL],9);
                if (mixed_precision) {
                    scaledType val;
                    val = int(request->arg(i).c_str());
                    float_output_weights[counterOL] = deScaleWeight(min_received_w, max_received_w, val);
                } else {
                    float_output_weights[counterOL] = request->arg(i).toFloat();
                }
                // Serial.print(F("received value OL = "));
                // Serial.println(request->arg(i).toFloat(),9);
                // Serial.print(F("new value OL = "));
                // Serial.println(float_output_weights[counterOL],9);
                counterOL++;
            }
        }
        // for (int i=0;i<args;i++)
        // {
        //     Serial.print("ARG[");
        //     Serial.print(request->argName(i).c_str());
        //     Serial.println("]: ");
        //     Serial.println(request->arg(i).c_str());
        // }
        request->send(200, "text/plain", "POST");
    }
}

void handleReceiveHiddenData(AsyncWebServerRequest *request)
{
	if (request->method() != HTTP_POST)
	{
		request->send(405, "text/plain", "Method Not Allowed");
	}
	else
	{

        if (debug_text){
            Serial.println(F("AsyncWebServer hidden layer msg received"));
        }

        //List all parameters (Compatibility)
        int args = request->args();
        // Serial.print("ARG[min_weight]: ");
        // Serial.println(request->arg("min_weight").c_str());
        // Serial.print("ARG[max_weight]: ");
        // Serial.println(request->arg("max_weight").c_str());
        // Serial.print("ARG[hidden_layer]: ");
        // Serial.println(request->arg("hidden_layer").c_str());

        //obtain the data from the FL server message
        float* float_hidden_weights = myNetwork.get_HiddenWeights();
        float min_received_w;
        float max_received_w;
        int batchNumber;
        int last;
        int batchSize;

        int counterHL = 0;
        for (int i=0;i<args;i++)
        {
            // Serial.print("ARG[");
            // Serial.print(request->argName(i).c_str());
            // Serial.print("]: ");
            // Serial.println(request->arg(i).c_str());
            if(request->argName(i) == "min_weight"){
                min_received_w = request->arg(i).toFloat();
            }
            else if(request->argName(i) == "min_weight"){
                max_received_w = request->arg(i).toFloat();
            }
            else if(request->argName(i) == "batchNumber"){
                batchNumber = request->arg(i).toInt();
            }
            else if(request->argName(i) == "last"){
                last = request->arg(i).toInt();
            }
            else if(request->argName(i) == "batchSize"){
                batchSize = request->arg(i).toInt();
            }
            else if(request->argName(i) == "hidden_layer"){
                // Serial.print(F("original value HL = "));
                // Serial.println(float_hidden_weights[counterHL],9);
                if (mixed_precision) {
                    scaledType val;
                    val = int(request->arg(i).c_str());
                    float_hidden_weights[i+batchNumber*batchSize] = deScaleWeight(min_received_w, max_received_w, val);
                } else {
                    float_hidden_weights[i+batchNumber*batchSize] = request->arg(i).toFloat();
                }
                // Serial.print(F("received value HL = "));
                // Serial.println(request->arg(i).toFloat(),9);
                // Serial.print(F("new value HL = "));
                // Serial.println(float_hidden_weights[counterHL],9);
                counterHL++;
            }
        }


        // for (int i=0;i<args;i++)
        // {
        //     Serial.print("ARG[");
        //     Serial.print(request->argName(i).c_str());
        //     Serial.println("]: ");
        //     Serial.println(request->arg(i).c_str());
        // }

        ////this way of checking if the whole model is received can make some devices 
        ////send their data for the next FL round before the current round is done, losing data. 
        // if (last == 1){
        //     waitingForFL = false;        
        //     Serial.print(F("Time to do a round of FL: "));
        //     unsigned long CurrentTime = millis();
        //     unsigned long ElapsedTime = CurrentTime - FLTime;
        //     Serial.print(ElapsedTime);
        //     Serial.println(F(" ms"));
        // }
        // Serial.print(F("waitingForFL = "));
        // Serial.println(waitingForFL);

        request->send(200, "text/plain", "POST");
    }
}
void handleReceiveFLConfirmation(AsyncWebServerRequest *request)
{   
    if (request->method() != HTTP_GET)
	{
		request->send(405, "text/plain", "Method Not Allowed");
	}
	else
	{
        if (debug_text){
            Serial.println(F("AsyncWebServer FL finished msg received"));
        }
        
        waitingForFL = false;        
        Serial.print(F("Time to do a round of FL: "));
        unsigned long CurrentTime = millis();
        unsigned long ElapsedTime = CurrentTime - FLTime;
        Serial.print(ElapsedTime);
        Serial.println(F(" ms"));
        
        // Serial.print(F("waitingForFL = "));
        // Serial.println(waitingForFL);

        // confirmation message for experimet_control.py
        Serial.println(F("Round of FL finished"));
        
        request->send(200, "text/plain", "OK");
    }
}

void handleNotFound(AsyncWebServerRequest *request)
{
	digitalWrite(LED_BUILTIN, LOW);

	String message = "File Not Found\n\n";

	message += "URI: ";
	//message += server.uri();
	message += request->url();
	message += "\nMethod: ";
	message += (request->method() == HTTP_GET) ? "GET" : "POST";
	message += "\nArguments: ";
	message += request->args();
	message += "\n";

	for (uint8_t i = 0; i < request->args(); i++)
	{
		message += " " + request->argName(i) + ": " + request->arg(i) + "\n";
	}

	request->send(404, "text/plain", message);
	digitalWrite(LED_BUILTIN, HIGH);
}
void initAsyncServer(){
    AsyncServer.on("/", HTTP_GET, [](AsyncWebServerRequest * request)
	{
		handleRoot(request);
	});

	AsyncServer.on("/sendMainData", HTTP_POST, [](AsyncWebServerRequest * request)
	{
		handleReceiveMainData(request);
	});

    AsyncServer.on("/sendOutputData", HTTP_POST, [](AsyncWebServerRequest * request)
	{
		handleReceiveOutputData(request);
	});	
    AsyncServer.on("/sendHiddenData", HTTP_POST, [](AsyncWebServerRequest * request)
	{
		handleReceiveHiddenData(request);
	});
    AsyncServer.on("/sendFLConfirmation", HTTP_GET, [](AsyncWebServerRequest * request)
	{
		handleReceiveFLConfirmation(request);
	});

	AsyncServer.onNotFound(handleNotFound);

	AsyncServer.begin();

	Serial.print(F("HTTP EthernetWebServer is @ IP : "));
	Serial.println(WiFi.localIP());
}


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
    initAsyncServer();
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
  if (client.connect(server, 5000)) {
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

    deviceIndex = readInt();

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

    if (waitingForFL) {
            //wait so the wifi port isn't being used
            delay(500);
            digitalWrite(LEDR, LOW);
            digitalWrite(LEDG, LOW);
            digitalWrite(LEDB, LOW);
            // Serial.println(F("waiting for FL"));
            // Serial.print(F("waitingForFL = "));
            // Serial.println(waitingForFL);
    }
    else if (Serial.available()) {
        char read = Serial.read();
        if (read == '>') {
            delay(50);
            digitalWrite(LEDR, LOW);
            digitalWrite(LEDG, LOW);
            digitalWrite(LEDB, LOW);
            //sendDataFL();
            // receiveDataFL();
        } else if (read == 't') {
            // Serial.println("main loop ");
            waitingForFL = true; 
            //set value first because if it's after sending the messages 
            //it could be set after receiving the data from the server, making an infinite loop.
            
            delay(deviceIndex*1000);
            // delay a different amount for each device to try to prevent sending the first FL message at the same time
            // the flask server blocks one of the devices if it receives 2 of the same message at the exact same time
            
            receiveSampleAndTrain();

            FLTime = millis();

            sendDataFL();
            sendOutputNodes();
            sendAllHiddenNodes();

            // Serial.print(F("Time to do a round of FL: "));
            // unsigned long CurrentTime = millis();
            // unsigned long ElapsedTime = CurrentTime - FLTime;
            // Serial.print(ElapsedTime);
            // Serial.println(F(" ms"));

            // delay(5000);
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
    if (debug_text){
        Serial.println(F("sendDataFL sending data..."));
    }
    while (client.available()){
        if (debug_text){
            Serial.println(F("ERROR: the client is already conected"));
        }
        char c = client.read();
        // Serial.print(c);
    }
    // Serial.println();
    client.stop();
    // if there's a successful connection:

    // compute the JSON before connecting to reduce the time spent connected
    // and to reuse it if the connection fails

    // Use arduinojson.org/v6/assistant to compute the capacity.
    const size_t capacity = JSON_OBJECT_SIZE(4) + 60;
    DynamicJsonDocument doc(capacity);
    // Serial.println(F("JSONdoc capacity:..."));
    // Serial.println(capacity);     
    doc["message"] = ">";
    doc["num_epochs"] = num_epochs;
    // Serial.println(F("computing min and max weights..."));
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
    doc["min_weight"] = (float) min_weight;
    doc["max_weight"] = (float) max_weight;
    int JSONsize = measureJsonPretty(doc);

    int tryCount = 0;
    bool retry = true;
    // WriteBufferingClient bufferedWifiClient{client, byteCapacity};
    while (retry and tryCount < 20){
        if (client.connect(server, 5000)) {
            retry = false;
            // client.setTimeout(10000);
            if (debug_text){
                Serial.println(F("connecting..."));
            }
            
            // Serial.println(F("wifi status: "));
            // Serial.println(WiFi.status());

            // WriteLoggingStream loggingClient(client, Serial);
            //send HTTP POST request for everything except the hiddenodes (because they don't fit in the document)
            client.println(F("POST /FL/sendData HTTP/1.1"));
            client.println("Host: " + stringHost);
            client.println(F("User-Agent: ArduinoWiFi/1.1"));
            client.println(F("Connection: close"));
            client.println(F("Content-Type: application/json"));
            
            client.print(F("Content-Length: "));
            client.println(JSONsize);
            // Terminate headers
            client.println(); //hay que dejar una linea vacía
            // Send body
            // Serial.println(F("sendDataFL Sending message"));
            unsigned long StartTime = millis();
            //optimized for speed
            // WriteBufferingClient bufferedWifiClient{client, byteCapacity};
            // Serial.println(F("sending JSON..."));
            serializeJsonPretty(doc, bufferedWifiClient);
            bufferedWifiClient.flush();
            // Serial.println(F("JSON sent"));
            // //regular version
            // serializeJsonPretty(doc, client);
            // // int stringSize = JSONdocString.length();
            // // Serial.print("Message length: ");
            // // Serial.println(stringSize);

            // Serial.println();
            
            unsigned long CurrentTime = millis();
            unsigned long ElapsedTime = CurrentTime - StartTime;
            // Serial.print(F("sendDataFL Data were sent successfully in: "));
            // Serial.print(ElapsedTime);
            // Serial.println(F(" ms"));

            // Serial.print(F("sendDataFL message weights: "));
            // Serial.print(doc.memoryUsage());
            // Serial.println(F(" bytes"));

            client.println();
            // //sending separate messages for the hiddenweights

            // wait for server response
            while (!client.available()){
            }
            // delay(responseDelay);
            if (debug_text){
                Serial.println(F("Server response received: "));
            }
            // 
            ////HTTP/1.1 201 CREATED -> successful message
            ////else connection error
            char responseCode[13] = "HTTP/1.1 201"; 
            int responseIt = 0;
            while (client.available()){
                char c = client.read();
                if (responseCode[responseIt] == '2'){
                    if (c != '2'){
                        retry = true;
                    }
                    // keep the while running to empty the recieved message buffer and close the connection properly.
                    responseIt++;
                }
                if (c == responseCode[responseIt]){
                    responseIt++;
                }
                // if (debug_text){
                //     Serial.print(c);
                // }
            }
            client.flush();
            client.stop();
            // if (debug_text){
            //     // Serial.println();
            // }

        } else {
            // if you couldn't make a connection:
            if (debug_text){
                Serial.println("connection failed");
            }
        }
        tryCount++;
        if (retry){
            if (debug_text){
                Serial.println(F("connection error. Trying again"));
            }
            // trying to debug the error that stops any comunication after a certain point
            // Serial.println(F("wifi status: "));
            // Serial.println(WiFi.status());
            if (tryCount > 3){
                // reset wifi connection
                status = WiFi.disconnect();
                delay(10000);
                while (status != WL_CONNECTED) {
                    if (debug_text){
                        Serial.print("Attempting to connect to SSID: ");
                        Serial.println(ssid);
                    }

                    // Connect to WPA/WPA2 network. Change this line if using open or WEP network:
                    status = WiFi.begin(ssid, pass);

                    // wait 10 seconds for connection:
                    delay(10000);
                }
            }
            delay(responseDelay);
        }
    }

// // close any connection before send a new request.
//   // This will free the socket on the WiFi shield
//   //for the AsyncServer
//   client.stop();
}

void sendHiddenNode(uint16_t batchNumber, uint16_t lastInBatch, uint16_t batchSize,
                    uint16_t start, uint16_t end, uint16_t numBatches)
                    {
    if (debug_text){
        Serial.println(F("sendHiddenNode sending data..."));
    }
    while (client.available()){
        if (debug_text){
            Serial.println(F("ERROR: the client is already conected"));
        }
        char c = client.read();
        // Serial.print(c);
    }
    client.stop();
    // Serial.println(F("connection stopped"));
    // if there's a successful connection:

    // compute the JSON before connecting to reduce the time spent connected
    // and to reuse it if the connection fails
    float* float_hidden_weights = myNetwork.get_HiddenWeights();
    float* float_output_weights = myNetwork.get_OutputWeights();
    size_t capacityHN = JSON_OBJECT_SIZE(4) + JSON_ARRAY_SIZE(batchSize) + 60;
    DynamicJsonDocument docHN(capacityHN);
    // Serial.print("the number of iterations is: ");
    // Serial.println(hiddenWeightsAmt);
    docHN["batchNumber"] = batchNumber;
    docHN["lastInBatch"] = lastInBatch;
    JsonArray deviceHiddenWeights = docHN.createNestedArray("HiddenWeights");
    docHN["batchSize"] = batchSize;
    docHN["numBatches"] = numBatches;
    char* hidden_weights = (char*) myNetwork.get_HiddenWeights();
            
    //serializeJsonPretty(docHN, Serial);
    for (uint16_t i = start; i < end; i++) {
        // Serial.print("start: ");
        // Serial.print(start);
        // Serial.print(" end: ");
        // Serial.println(end);
        // Serial.print("iteration: ");
        // Serial.println(i);
                
        //serializeJsonPretty(docHN, Serial);
        deviceHiddenWeights.add(float_hidden_weights[i]);
    }
    int JSONsize = measureJsonPretty(docHN);

    int tryCount = 0;
    bool retry = true;
    // WriteBufferingClient bufferedWifiClient{client, byteCapacity};
    while (retry and tryCount < 10){
        if (client.connect(server, 5000)) {
            // client.setTimeout(500);
            retry = false;
            if (debug_text){
                Serial.println(F("connecting..."));
            }
            
            // Serial.println(F("wifi status: "));
            // Serial.println(WiFi.status());
            //send message when all is obtained
            // Serial.println(F("Preparing message "));

            client.println(F("POST /FL/sendHiddenNodeBatch HTTP/1.1"));
            client.println("Host: " + stringHost);
            client.println(F("User-Agent: ArduinoWiFi/1.1"));
            client.println(F("Connection: close"));
            client.println(F("Content-Type: application/json"));
            
            if (debug_text){
                Serial.print(F("Sending message "));
                Serial.println(batchNumber);
            }

            client.print(F("Content-Length: "));
            client.println(JSONsize);
            // Terminate headers
            client.println(); //hay que dejar una linea vacía
            // Send body
            //optimized for speed
            // WriteBufferingClient bufferedWifiClient{client, byteCapacity};
            
            // Serial.println(F("sending JSON..."));
            serializeJsonPretty(docHN, bufferedWifiClient);
            bufferedWifiClient.flush();
            // Serial.println(F("JSON sent"));
            
            // serializeJsonPretty(docHN, client);
            // //serializeJsonPretty(docHN, Serial);
                        
            // wait for server response
            while (!client.available()){
            }

            // delay(responseDelay);
            if (debug_text){
                Serial.println(F("Server response received: "));
            }
            ////HTTP/1.1 201 CREATED -> successful message
            ////else connection error
            char responseCode[13] = "HTTP/1.1 201"; 
            int responseIt = 0;
            while (client.available()){
                char c = client.read();
                if (responseCode[responseIt] == '2'){
                    if (c != '2'){
                        retry = true;
                    }
                    // keep the while running to empty the recieved message buffer and close the connection properly.
                    responseIt++;
                }
                if (c == responseCode[responseIt]){
                    responseIt++;
                }
                // Serial.print(c);
            }
            client.flush();
            client.stop();
            // Serial.println();
        } else {
            // if you couldn't make a connection:
            if (debug_text){
                Serial.println(F("Connection failed"));
            }
        }
        tryCount++;
        if (retry){
            if (debug_text){
                Serial.println(F("Connection error. Trying again"));
            }
            // Serial.println(F("wifi status: "));
            // Serial.println(WiFi.status());
            if (tryCount > 3){
                // reset wifi connection
                status = WiFi.disconnect();
                delay(10000);
                while (status != WL_CONNECTED) {
                    if (debug_text){
                        Serial.print("Attempting to connect to SSID: ");
                        Serial.println(ssid);
                    }
                    // Connect to WPA/WPA2 network. Change this line if using open or WEP network:
                    status = WiFi.begin(ssid, pass);

                    // wait 10 seconds for connection:
                    delay(10000);
                }
            }
            delay(responseDelay);
        }
    }
//     // close any connection before send a new request.
//   // This will free the socket on the WiFi shield
//   //for the AsyncServer
//   client.stop();
}

void sendAllHiddenNodes(){
// close any connection before send a new request.
    //sending separate messages for the hiddenweights
    if (debug_text){
        Serial.println(F("computing HiddenWeights..."));
    }
    
    unsigned long StartTime = millis();
    uint16_t batchSize = 500;
    uint16_t batchNumber = 0;
    uint16_t numBatches = (hiddenWeightsAmt + (batchSize - 1)) / batchSize;
    if (debug_text){
        Serial.print("total hidden neuron weights: ");
        Serial.println(hiddenWeightsAmt);
        Serial.print("number of hidden neuron weights batches: ");
        Serial.println(numBatches);
    }
    for (uint16_t i = 0; i < hiddenWeightsAmt; i = i + batchSize) {
        if (debug_text){
            Serial.print("All iteration: ");
            Serial.println(i);
        }
        if (i + batchSize > hiddenWeightsAmt){
            if (debug_text){
                Serial.println("last hidden weight batch");
            }
            sendHiddenNode(batchNumber, hiddenWeightsAmt-i, batchSize, i, hiddenWeightsAmt, numBatches);
        }
        else{
            if (debug_text){
                Serial.println("not last hidden weight batch");
            }
            sendHiddenNode(batchNumber, batchSize, batchSize, i, i + batchSize, numBatches);
            }
            //client.stop();
            if (debug_text){
                Serial.println("batch sent");
            }
            batchNumber++;
            //delay(10000);
        }

    unsigned long CurrentTime = millis();
    unsigned long ElapsedTime = CurrentTime - StartTime;
    if (debug_text){
        Serial.print(F("HiddenWeights Data were sent successfully in: "));
        Serial.print(ElapsedTime);
        Serial.println(F(" ms"));
    }
}

void sendOutputNodes(){
    if (debug_text){
        Serial.println(F("sendOutputNodes sending data..."));
    }
    while (client.available()){
        if (debug_text){
            Serial.println(F("ERROR: the client is already conected"));
        }
        char c = client.read();
        // Serial.print(c);
    }
    client.stop();
    // Serial.println(F("connection stopped"));
    // if there's a successful connection:

    // compute the JSON before connecting to reduce the time spent connected
    // and to reuse it if the connection fails
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

    size_t capacityON = JSON_ARRAY_SIZE(outputWeightsAmt) + 60;
    DynamicJsonDocument docON(capacityON);
    // Serial.println("computing OutputWeights...");
    JsonArray deviceOutputWeights = docON.createNestedArray("OutputWeights");
    // Sending output layer
    char* output_weights = (char*) myNetwork.get_OutputWeights();
    for (uint16_t i = 0; i < outputWeightsAmt; i++) {
        if (mixed_precision) {
            scaledType weight = scaleWeight(min_weight, max_weight, float_output_weights[i]);
            scaledType casted = weight;
            //Serial.write((byte*) &casted, sizeof(scaledType));
            deviceOutputWeights.add(casted);
        } else {
            //Serial.write((byte*) &float_output_weights[i], sizeof(float)); // debug
            deviceOutputWeights.add(float_output_weights[i]);
        }
    }
    int JSONsize = measureJsonPretty(docON);

    int tryCount = 0;
    bool retry = true;
    // WriteBufferingClient bufferedWifiClient{client, byteCapacity};
    while (retry and tryCount < 10){
        if (client.connect(server, 5000)) {
            // client.setTimeout(500);
            retry = false;
            if (debug_text){
                Serial.println(F("connecting..."));
            }

            // Serial.println(F("wifi status: "));
            // Serial.println(WiFi.status());
            //send message when all is obtained
            // Serial.println(F("Preparing message "));

            client.println(F("POST /FL/sendOutputNodes HTTP/1.1"));
            client.println("Host: " + stringHost);
            client.println(F("User-Agent: ArduinoWiFi/1.1"));
            client.println(F("Connection: close"));
            client.println(F("Content-Type: application/json"));
            client.print(F("Content-Length: "));
            client.println(JSONsize);
            // Terminate headers
            client.println(); //hay que dejar una linea vacía
            // Send body
            //optimized for speed
            // WriteBufferingClient bufferedWifiClient{client, byteCapacity};
            
            // Serial.println(F("sending JSON..."));
            serializeJsonPretty(docON, bufferedWifiClient);
            bufferedWifiClient.flush();
            // Serial.println(F("JSON sent"));
            
            // serializeJsonPretty(docHN, client);
            // serializeJsonPretty(docON, Serial);
            
            // wait for server response
            while (!client.available()){
            }
            // delay(responseDelay);
            if (debug_text){
                Serial.println(F("Server response received: "));
            }
            ////HTTP/1.1 201 CREATED -> successful message
            ////else connection error
            char responseCode[13] = "HTTP/1.1 201"; 
            int responseIt = 0;
            while (client.available()){
                char c = client.read();
                if (responseCode[responseIt] == '2'){
                    if (c != '2'){
                        retry = true;
                    }
                    // keep the while running to empty the recieved message buffer and close the connection properly.
                    responseIt++;
                }
                if (c == responseCode[responseIt]){
                    responseIt++;
                }
                // Serial.print(c);
            }
            client.flush();
            client.stop();
            // Serial.println();
        } else {
            // if you couldn't make a connection:
            if (debug_text){
                Serial.println(F("connection failed"));
            }
        }
        tryCount++;
        if (retry){
            if (debug_text){
                Serial.println(F("connection error. Trying again"));
            }
            // Serial.println(F("wifi status: "));
            // Serial.println(WiFi.status());
            if (tryCount > 3){
                // reset wifi connection
                status = WiFi.disconnect();
                delay(10000);
                while (status != WL_CONNECTED) {
                    if (debug_text){
                        Serial.print("Attempting to connect to SSID: ");
                        Serial.println(ssid);
                    }
                    // Connect to WPA/WPA2 network. Change this line if using open or WEP network:
                    status = WiFi.begin(ssid, pass);

                    // wait 10 seconds for connection:
                    delay(10000);
                }
            }
            delay(responseDelay);
        }
    }
//     // close any connection before send a new request.
//   // This will free the socket on the WiFi shield
//   //for the AsyncServer
//   client.stop();
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