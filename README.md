# TinyML with Arduino Portenta H7

## Introduction
This project consists in doing FL on Portenta H7 boards using an external FL server, and doing the connection with wifi.

This project has three parts:
### 1. Arduino Portenta H7

Training the models, sending them to the FL server and receiving the FL model. Contained in the src folder.
### 2. FL server:

Receiving the models from the boards, computing the FL model and sending it to all the boards. It's done using Flask. Contained in FL_server_part.py.
### 3. Experiment control:

Sending the blank models and samples to the boards. Contained in experiment_control.py.

## Necessary changes
These changes are necessary to be able to execute this project on your machine.

### Setting up the wifi connection:
In src/main.ino change the values of the variables ssid and pass to the ones of your wifi network:

```C
char ssid[] = "yourSSID";   // your network SSID (name)
char pass[] = "yourPASS";   // your network password
```

### Connecting Portenta to Flask
In src/main.ino change the values of the variables server and stringHost to the IP of the Flask server in FL_server_part.py.
The Flask server will print it's IP when it's executed, but only write the numbers on the variables, without http:// or the port.
The port is introduced when declaring the ``WiFiServer``.
```C
// server address: IP address of the flask server
IPAddress server(999,999,999,999);
String stringHost = "999.999.999.999";
```

## Testing

### Defining the number of boards of the experiment
In FL_server_part.py change ``self.numdevices`` to the number of boards of your test.

### Setting values for the model
The number of InputNodes, HiddenNodes and OutputNodes are defined in the code in three files:

src/neural_network.h:
```C
static const int InputNodes = 650;
static const int HiddenNodes = 25;
static const int OutputNodes = 4;
```

FL_server_part.py:
```Python
input_nodes = 650
output_nodes = len(keywords_buttons)
size_hidden_nodes = 25
```

experiment_control.py:
```Python
output_nodes = len(keywords_buttons)
test_samples_amount = 60
size_hidden_nodes = 25
size_hidden_layer = (650+1)*size_hidden_nodes
```

The values have to be the same in all three files.

### Modifying the wifi transmission speed
By default the wifi client writes byte per byte. Using the library [StreamUtils](https://github.com/bblanchon/ArduinoStreamUtils) you can change it to write in chunks of bytes.
In src/main.ino change the value of ``byteCapacity`` to change the size of the chunks of bytes of all wifi messages. Bigger chunks increase the speed.

## Instructions
In order to run this project:
1. Flash the sketch in src/main.ino into all the boards
2. Execute FL_server_part.py to start running the Flask FL server
3. While the server is running, execute experiment_control.py to initialize the models and send the samples, introducing the number of devices and their ports at the start of the execution. The boards will begin training and communicating with the Flask server to do FL.

## Known issues

### error: reference to '__FlashStringHelper' is ambiguous 
The full error is:
.pio/libdeps/portenta_h7_m7/StreamUtils/src/StreamUtils/Streams/ProgmemStream.hpp:22:23: error: reference to '__FlashStringHelper' is ambiguous
   ProgmemStream(const __FlashStringHelper* ptr)
   
If this happens when you try to compile the project, using PlatformIO open the file ProgmemStream.hpp in the folder .pio/libdeps/portenta_h7_m7/StreamUtils/src/StreamUtils/Streams/ of the project, and comment the lines 22 and 23:

```C
class ProgmemStream : public Stream {
 public:
  ProgmemStream(const void* ptr, size_t size)
      : _ptr(reinterpret_cast<const char*>(ptr)), _size(size) {}

  ProgmemStream(const char* ptr) : _ptr(ptr), _size(ptr ? strlen_P(ptr) : 0) {}

  // ProgmemStream(const __FlashStringHelper* ptr)
  //     : ProgmemStream{reinterpret_cast<const char*>(ptr)} {}
```
Then the project should compile correctly.