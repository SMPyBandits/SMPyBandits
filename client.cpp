/**
    C++ client example using sockets
    http:// www.binarytides.com/code-a-simple-socket-client-class-in-c/
*/
#include <cstdlib>         // rand
#include <iostream>        // cout
#include <stdio.h>         // printf
#include <string.h>        // strlen
#include <string>          // string
#include <sys/socket.h>    // socket
#include <arpa/inet.h>     // inet_addr
#include <netdb.h>         // hostent
#include <thread>          // sleep
#include <chrono>          // milliseconds

using namespace std;

/**
    TCP Client class
*/
class tcp_client {
    private:
        int sock;
        std::string address;
        int port;
        struct sockaddr_in server;

    public:
        tcp_client();
        bool conn(string, int);
        bool send_data(string data);
        string receive(int);
};

tcp_client::tcp_client() {
    sock = -1;
    port = 0;
    address = "";
}

/**
    Connect to a host on a certain port number
*/
bool tcp_client::conn(string address , int port) {
    // create socket if it is not already created
    if (sock == -1) {
        // Create socket
        sock = socket(AF_INET , SOCK_STREAM , 0);
        if (sock == -1) {
            perror("Could not create socket");
        }

        cout << "Socket created\n";
    }
    else    {   /* OK , nothing */  }

    // setup address structure
    if (inet_addr(address.c_str()) == -1) {
        struct hostent *he;
        struct in_addr **addr_list;

        // resolve the hostname, its not an ip address
        if ( (he = gethostbyname( address.c_str() ) ) == NULL) {
            // gethostbyname failed
            herror("gethostbyname");
            cout << "Failed to resolve hostname\n";

            return false;
        }

        // Cast the h_addr_list to in_addr , since h_addr_list also has the ip address in long format only
        addr_list = (struct in_addr **) he->h_addr_list;

        for(int i = 0; addr_list[i] != NULL; i++) {
            server.sin_addr = *addr_list[i];

            cout << address << " resolved to " << inet_ntoa(*addr_list[i]) << endl;

            break;
        }
    }

    // plain ip address
    else {
        server.sin_addr.s_addr = inet_addr( address.c_str() );
    }

    server.sin_family = AF_INET;
    server.sin_port = htons( port );

    // Connect to remote server
    if (connect(sock , (struct sockaddr *)&server , sizeof(server)) < 0) {
        perror("connect failed. Error");
        return 1;
    }

    cout << "Connected\n";
    return true;
}

/**
    Send data to the connected host
*/
bool tcp_client::send_data(string data) {
    // Send some data
    if ( send(sock , data.c_str() , strlen( data.c_str() ) , 0) < 0) {
        perror("Send failed : ");
        return false;
    }
    std::printf("\nData %s send\n", data.c_str());

    return true;
}

/**
    Receive data from the connected host
*/
string tcp_client::receive(int size=1) {
    char buffer[size];
    string reply;

    // Receive a reply from the server
    if ( recv(sock , buffer , sizeof(buffer) , 0) < 0)
    {
        puts("recv failed");
    }

    reply = buffer;
    return reply;
}

int main(int argc , char *argv[]) {
    std::srand(std::time(0)); // use current time as seed for random generator

    float reward;
    tcp_client c;
    string received;

    // connect to host
    // TODO read this from command line
    c.conn("0.0.0.0", 10000);

    // send some data
    c.send_data("Hi!");

    // receive and echo reply
    while (true) {
        received = c.receive();
        std::printf("\nReceived '%s'...", received.c_str());
        // send some data
        reward = rand() / static_cast<float>(RAND_MAX);
        c.send_data(std::to_string(reward));
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    };

    // done
    return 0;
}