// #! g++ -Wall -Iinclude -o env_client.exe include/docopt.cpp env_client.cpp
/**
    C++ client, using sockets, to simulate a MAB environment.
    So far, only Bernoulli arms are supported.

    - Author: Lilian Besson
    - License: MIT License (https://lbesson.mit-license.org/)
    - Date: 28-07-2017
    - Online: https://smpybandits.github.io/
    - Reference: http://www.binarytides.com/code-a-simple-socket-client-class-in-c/
*/

// Include libraries
#include <arpa/inet.h>  // inet_addr
#include <chrono>       // milliseconds
#include <cstdlib>      // rand
#include <docopt.h>     // docopt command line parser
#include <iostream>     // streams, <<, >>
#include <netdb.h>      // hostent
#include <stdio.h>      // printf
#include <string.h>     // strlen
#include <string>       // string
#include <sys/socket.h> // socket
#include <thread>       // sleep

// No need for std::printf, std::string etc
using namespace std;

/**
    TCP Client class
*/
class tcp_client {
    private:
        int sock;
        string address;
        int port;
        struct sockaddr_in server;

    public:
        tcp_client();
        bool conn(string, int);
        bool send_data(string data);
        string receive(int);
};

/**
    Default initializer
*/
tcp_client::tcp_client() {
    sock = -1;
    port = 0;
    address = "";
}

/**
    Connect to a host on a certain port number
*/
bool tcp_client::conn(string address, int port) {
    const char *c_address;
    c_address = address.c_str();

    // create socket if it is not already created
    if (sock == -1) {
        // Create socket
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == -1) {
            perror("Could not create socket");
        }
        printf("Socket created\n");
    } else {
        /* OK , nothing */
    }

    // setup address structure
    if (inet_addr(c_address) < 0) {
        struct hostent *he;
        struct in_addr **addr_list;

        // resolve the hostname, its not an ip address
        if ((he = gethostbyname(c_address)) == NULL) {
            // gethostbyname failed
            herror("gethostbyname");
            printf("Failed to resolve hostname for '%s'... Try again please.\n",
                    c_address);
            return false;
        }

        // Cast the h_addr_list to in_addr , since h_addr_list also has the ip
        // address in long format only
        addr_list = (struct in_addr **)he->h_addr_list;

        for (int i = 0; addr_list[i] != NULL; i++) {
            server.sin_addr = *addr_list[i];
            printf("Address '%s' resolved to '%s'...\n", c_address, inet_ntoa(*addr_list[i]));
            break;
        }
    }
    // plain ip address
    else {
        server.sin_addr.s_addr = inet_addr(c_address);
    }

    server.sin_family = AF_INET;
    server.sin_port = htons(port);

    // Connect to remote server
    if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0) {
        perror("Connect failed. Error!");
        return false;
    }

    printf("Connected to '%s' with port '%d' !\n", c_address, port);
    return true;
}

/**
    Send data to the connected host
*/
bool tcp_client::send_data(string data) {
    // Send some data
    if (send(sock, data.c_str(), strlen(data.c_str()), 0) < 0) {
        perror("Send failed : ");
        return false;
    }
    printf("\nData '%s' successfully sent!\n", data.c_str());
    return true;
}

/**
    Receive data from the connected host
*/
string tcp_client::receive(int size = 4) {
    char buffer[size];
    string reply;

    // Receive a reply from the server
    if (recv(sock, buffer, sizeof(buffer), 0) < 0) {
        printf("recv failed...");
    }

    reply = buffer;
    return reply;
}

// Macro to have a random float in [0, 1)
#define random_float() (rand() / static_cast<float>(RAND_MAX))

/**
    Draw one sample from a Bernoulli distribution of a certain mean.
*/
float bernoulli_draw(float mean) {
    // send some data, random in [0, 1]
    if (random_float() < mean) {
        return 1;
    } else {
        return 0;
    }
}

/**
    Infinite loop, sending random rewards on the asked arm.

    - create the socket and connect,
    - continuously read the socket for a channel id number #i,
    - generate a random reward for mean = mu[i],
    - send back that reward to the socket.
*/
int loop(
    string address,
    int port,
    vector<float> means,
    int milli_sleep = 2000
) {
    srand(time(0)); // use current time as seed for random generator

    tcp_client c;
    string received;
    float reward;
    int channel;

    c.conn(address, port); // connect to host

    // send some data, just a stupid useless handcheck
    c.send_data("Hi from env_client.exe !");

    while (true) { // receive and echo reply
        try {
            received = c.receive();
            try {
                channel = stoi(received);
            } catch (const invalid_argument &) {
                channel = 0;
            }
            printf("\nReceived '%s' = channel #'%d'...", received.c_str(), channel);
            reward = bernoulli_draw(means[channel]);
            c.send_data(to_string(reward));
        } catch (const invalid_argument &) {
            printf("\nReceived something not correctly understood by stoi(), no problem we continue.");
        } catch (const runtime_error &) {
            printf("\nRuntime error in the loop, no issue we continue.");
        }
        this_thread::sleep_for(chrono::milliseconds(milli_sleep));
    };

    return 0; // done
}

/**
    Convert a string, read from the cli, to a vector of float number.
    In Python, that would be map(float, arr.split(',')), but it takes 15 lines
    here. Yay! Finally debugged, OK.
*/
vector<float> array_from_str(string arr) {
    // cout << "arr = " << arr << endl;  // DEBUG
    uint nb = 1;
    uint size_arr = arr.size();
    // cout << "size_arr = " << size_arr << endl;  // DEBUG
    // first, find nb
    uint index = 0;
    uint found;
    while (true) {
        found = arr.find(',', index);
        // cout << "\nfound = " << found << endl;  // DEBUG
        // cout << "index = " << index << endl;  // DEBUG
        if ((found == string::npos) || (found >= size_arr)) {
            break;
        } else {
            nb += 1;
            index = found + 1;
        }
    }
    // cout << "nb = " << nb << endl;  // DEBUG

    // allocate vector
    vector<float> means;
    // cout << "\nSecond step...\n";  // DEBUG

    // then iterate on the string, convert and store
    uint i = 0;
    index = 0;
    while (true) {
        found = arr.find(',', index);
        // cout << "\nfound = " << found << endl;  // DEBUG
        // cout << "index = " << index << endl;  // DEBUG
        if (i > nb) {
            break;
        } else {
            // cout << "reading = " << stof(arr.substr(index, found)) << endl;  // DEBUG
            means.push_back(stof(arr.substr(index, found)));
            // cout << "means[i] = " << means[i] << endl;  // DEBUG
            index = found + 1;
            i += 1;
            // cout << "i = " << i << endl;  // DEBUG
        }
    }
    return means;
}

/**
    documentation for the cli generated with docopt
*/
static const char USAGE[] =
    R"(C++ Client to play multi-armed bandits problem against.

Usage:
    env_client.exe [--port=<PORT>] [--host=<HOST>] [--speed=<SPEED>] [<bernoulli_means>]
    env_client.exe (-h|--help)
    env_client.exe --version

Options:
    -h --help   Show this screen.
    --version   Show version.
    --port=<PORT>   Port to use for the TCP connection [default: 10000].
    --host=<HOST>   Address to use for the TCP connection [default: 0.0.0.0].
    --speed=<SPEED>   Speed of emission in milliseconds [default: 1000].
)";

/**
    Main function, parsing the cli arguments with docopt::docopt and calling
    loop() with the good arguments.
*/
int main(
    int argc,
    const char **argv
) {
    string address;
    long port, speed;
    vector<float> default_means = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    vector<float> means;

    // parse the cli arguments, magically with docopt::docopt
    map<string, docopt::value> args =
        docopt::docopt(USAGE, {argv + 1, argv + argc},
                        true,                             // show help if requested
                        "MAB environment C++ client v0.1" // version string
        );

    address = args["--host"].asString();
    port = args["--port"].asLong();
    speed = args["--speed"].asLong();
    if (args["<bernoulli_means>"].isString()) {
        printf("Bernoulli means = '%s'\n", args["<bernoulli_means>"].asString());
        means = array_from_str(args["<bernoulli_means>"].asString());
    } else {
        means = default_means;
    }

    cout << "- address = " << address << endl; // DEBUG
    cout << "- port = " << port << endl;       // DEBUG
    cout << "- speed = " << speed << endl;     // DEBUG
    for (uint i = 0; i < means.size(); i++) {
        cout << "- means[" << i << "] = " << means[i] << endl; // DEBUG
    }
    cout << endl << "Calling loop... starting..." << endl; // DEBUG
    return loop(address, port, means, speed);
}