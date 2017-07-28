// #! g++ -Wall -o env_client.exe env_client.cpp
/**
    C++ client, using sockets, to simulate a MAB environment.

    - Author: Lilian Besson
    - License: MIT License (https://lbesson.mit-license.org/)
    - Date: 28-07-2017
    - Online: http://banditslilian.gforge.inria.fr/
    - Reference: http:// www.binarytides.com/code-a-simple-socket-client-class-in-c/
*/

// Include libraries
#include <cstdlib>         // rand
#include <stdio.h>         // printf
#include <cstdio>          // scanf ?
#include <string.h>        // strlen
#include <string>          // string
#include <sys/socket.h>    // socket
#include <arpa/inet.h>     // inet_addr
#include <netdb.h>         // hostent
#include <thread>          // sleep
#include <chrono>          // milliseconds
#include <docopt.h>        // docopt command line parser

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
    const char* c_address;
    c_address = address.c_str();

    // create socket if it is not already created
    if (sock == -1) {
        // Create socket
        sock = socket(AF_INET , SOCK_STREAM , 0);
        if (sock == -1) {
            perror("Could not create socket");
        }
        printf("Socket created\n");
    }
    else    {   /* OK , nothing */  }

    // setup address structure
    if (inet_addr(c_address) == -1) {
        struct hostent *he;
        struct in_addr **addr_list;

        // resolve the hostname, its not an ip address
        if ( (he = gethostbyname(c_address) ) == NULL) {
            // gethostbyname failed
            herror("gethostbyname");
            printf("Failed to resolve hostname for '%s'... Try again please.\n", c_address);
            return false;
        }

        // Cast the h_addr_list to in_addr , since h_addr_list also has the ip address in long format only
        addr_list = (struct in_addr **) he->h_addr_list;

        for (int i = 0; addr_list[i] != NULL; i++) {
            server.sin_addr = *addr_list[i];
            printf("Address '%s' resolved to '%s'...\n", c_address, inet_ntoa(*addr_list[i]));
            break;
        }
    }
    // plain ip address
    else {
        server.sin_addr.s_addr = inet_addr( c_address );
    }

    server.sin_family = AF_INET;
    server.sin_port = htons(port);

    // Connect to remote server
    if (connect(sock , (struct sockaddr *)&server , sizeof(server)) < 0) {
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
    if ( send(sock , data.c_str(), strlen( data.c_str() ), 0) < 0) {
        perror("Send failed : ");
        return false;
    }
    printf("\nData '%s' send\n", data.c_str());
    return true;
}

/**
    Receive data from the connected host
*/
string tcp_client::receive(int size=16) {
    char buffer[size];
    string reply;

    // Receive a reply from the server
    if ( recv(sock , buffer , sizeof(buffer) , 0) < 0) {
        printf("recv failed...");
    }

    reply = buffer;
    return reply;
}

#define random_float()  rand() / static_cast<float>(RAND_MAX);

int loop(string address, int port, float* means, int mssleep=2000) {
    srand(time(0)); // use current time as seed for random generator

    tcp_client c;
    string received;
    float bern;
    int reward;
    int channel;

    // connect to host
    c.conn(address, port);

    // send some data, just a stupid useless handcheck
    c.send_data("Hi!");

    // receive and echo reply
    while (true) {
        received = c.receive();
        printf("\nReceived '%s'...", received.c_str());
        channel = stoi(received);
        printf("\n  = Channel '%d'...", channel);
        // send some data, random in [0, 1]
        bern = random_float();
        if (bern < means[channel]) {
            reward = 1;
        } else {
            reward = 0;
        }
        c.send_data(to_string(reward));
        this_thread::sleep_for(chrono::milliseconds(mssleep));
    };

    // done
    return 0;
}

static const char USAGE[] =
R"(C++ Client to play multi-armed bandits problem against.

Usage:
    env_client.exe [--port=<PORT>] [--host=<HOST>] [--speed=<SPEED>]
    env_client.exe (-h|--help)
    env_client.exe --version

Options:
    -h --help   Show this screen.
    --version   Show version.
    --port=<PORT>   Port to use for the TCP connection [default: 10000].
    --host=<HOST>   Address to use for the TCP connection [default: 0.0.0.0].
    --speed=<SPEED>   Speed of emission in milliseconds [default: 1000].
)";

int main(int argc , const char** argv) {
    string address;
    long port;
    long speed;

    map<string, docopt::value> args
        = docopt::docopt(USAGE,
                         { argv + 1, argv + argc },
                         true,               // show help if requested
                         "MAB environment C++ client v0.1"  // version string
    );

    address = args["--host"].asString();
    port = args["--port"].asLong();
    speed = args["--speed"].asLong();

    // TODO read this from command line
    float means[] = { 0., 0., 0., 0., 0., 0., 0., 0.7, 0.8, 0.9 };
    return loop(address, port, means, speed);
}