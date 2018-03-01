// #! g++ -std=c++11 -Iinclude -o test_sub.exe test_sub.cpp -pthread
/**
    Test of https://github.com/arun11299/cpp-subprocess

    - Author: Lilian Besson
    - License: MIT License (https://lbesson.mit-license.org/)
    - Date: 09-08-2017
    - Online: https://smpybandits.github.io/
    - Reference: https://github.com/arun11299/cpp-subprocess
*/

// Include libraries
#include <iostream>        // streams, <<, >>
#include <string.h>        // strlen
#include <string>
#include "subprocess.hpp"  // From https://github.com/arun11299/cpp-subprocess

// Macros to send a message
#define send(msg)          p.send(msg, strlen(msg))
#define communicate(msg)   p.communicate(msg, strlen(msg))


int main() {
    namespace sp = subprocess;

    // auto p = sp::Popen({"python3"}, sp::input{sp::PIPE});
    auto p = sp::Popen({"python3"}, sp::input{sp::PIPE}, sp::output{sp::PIPE});
    auto input  = p.input();
    auto output = p.output();

    // Import all the policies
    send("from Policies import *\n");
    // std::cout << output.buf.data() << std::endl;

    // Create the policy
    send("policy = UCBalpha(10, alpha=0.5)\n");
    // std::cout << output.buf.data() << std::endl;

    // Print it
    send("print(policy)\n");
    // std::cout << output.buf.data() << std::endl;

    // Print it
    send("print(policy)\n");
    // std::cout << output.buf.data() << std::endl;

    return 0;
}
