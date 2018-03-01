#include <string>
#include <iostream>
#include <boost/python.hpp>

using namespace boost::python;

int main(int, char **) {
    Py_Initialize();

    try {
        object module = import("__main__");
        object name_space = module.attr("__dict__");
        exec_file("test.py", name_space, name_space);

        object choice = name_space["choice"];
        object result = choice();
        // result is a dictionary
        std::string val = extract<std::string>(result["val"]);
        std::cout << val << std::endl;
    }
    catch (error_already_set) {
        PyErr_Print();
    }

    Py_Finalize();
    return 0;
}