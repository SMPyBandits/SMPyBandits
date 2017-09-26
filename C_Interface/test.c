// Compile with
// $ gcc -I/usr/include/python3.5 test.c -lpython3.5
// Execute with
// $ ./a.out Arms uniformMeansWithSparsity 10 3

// depending on distros, the exact path or Python version may vary.
#include </usr/include/python3.5/Python.h>
#include <stdlib.h>
#include <string.h>
// #include <time.h>

// Initialize rand() "safely"
// srand(time(NULL));

int loop(PyObject *pPolicy, PyObject *pMAB, int horizon) {
    int t;
    PyObject *pChoice, *pReward, *pArguments;
    pArguments = PyTuple_New(2);

    for (t = 0; t < horizon; t++) {
        // Ask choice
        pChoice = PyObject_CallObject(PyObject_GetAttrString(pPolicy, "choice"), NULL);
        printf("At time t = '%d' the policy 'choice' method suggested to use '%s'\n", t, PyBytes_AsString(PyObject_Str(pChoice)));
        // Generate reward
        pReward = PyObject_CallObject(PyObject_GetAttrString(pMAB, "draw"), pChoice);
        printf("    Observed reward '%s'\n", PyBytes_AsString(PyObject_Str(pReward)));
        // Send feedback
        PyTuple_SetItem(pArguments, 0, pChoice);
        PyTuple_SetItem(pArguments, 1, pReward);
        PyObject_CallObject(PyObject_GetAttrString(pPolicy, "getReward"), pArguments);
        printf("    Gave back feedback to method 'getReward' with arguments '%s'\n", PyBytes_AsString(PyObject_Str(pArguments)));
    }

    return 0;
}

int main(int argc, char *argv[]) {
    // PyObject *pName, *pModule, *pFunc, *pValue;
    int nbArms = 3;
    int horizon = 100;
    double alpha = 0.5;

    Py_Initialize();
    // add local path to the sys.path
    PyList_Append(PyObject_GetAttrString(PyImport_ImportModule("sys"), "path"), PyBytes_FromString("."));
    PyList_Append(PyObject_GetAttrString(PyImport_ImportModule("sys"), "path"), PyBytes_FromString(".."));
    // See https://stackoverflow.com/a/8859538/

    PyObject *pMAB_class;
    pMAB_class = PyObject_GetAttrString(PyImport_ImportModule("Environment"), "MAB");
    printf("   'pMAB_class' = '%s'\n", PyBytes_AsString(PyObject_Str(pMAB_class)));
    PyObject *pBernoulli_class;
    pBernoulli_class = PyObject_GetAttrString(PyImport_ImportModule("Arms"), "Bernoulli");
    printf("   'pBernoulli_class' = '%s'\n", PyBytes_AsString(PyObject_Str(pBernoulli_class)));

    PyObject *pArguments;
    pArguments = PyDict_New();
        PyDict_SetItemString(pArguments, "arm_type", pBernoulli_class);
        PyObject *pParams;
        pParams = PyList_New(nbArms);
        // FIXME do this more generically!
            PyList_Append(pParams, PyFloat_FromDouble(0.1));
            PyList_Append(pParams, PyFloat_FromDouble(0.5));
            PyList_Append(pParams, PyFloat_FromDouble(0.9));
            printf("   'pParams' = '%s'\n", PyBytes_AsString(PyObject_Str(pParams)));
        PyDict_SetItemString(pArguments, "params", pParams);
        // configuration = {
        //     'arm_type': Bernoulli,
        //     'params':   [0.1, 0.5, 0.9]
        // }
    printf("   'pArguments' = '%s'\n", PyBytes_AsString(PyObject_Str(pArguments)));
    PyObject *pMAB;
    pMAB = PyObject_CallObject(pMAB_class, pArguments);
    printf("   'pMAB' = '%s'\n", PyBytes_AsString(PyObject_Str(pMAB)));

    PyObject *pUCB_class;
    pUCB_class = PyObject_GetAttrString(PyImport_ImportModule("Policies"), "UCBalpha");
    printf("   'pUCB_class' = '%s'\n", PyBytes_AsString(PyObject_Str(pUCB_class)));

    PyObject *pUCB;
    pArguments = PyTuple_New(2);
    PyTuple_SetItem(pArguments, 0, PyLong_FromLong(nbArms));
    PyTuple_SetItem(pArguments, 1, PyFloat_FromDouble(alpha));
    printf("   'pArguments' = '%s'\n", PyBytes_AsString(PyObject_Str(pArguments)));
    pUCB = PyObject_CallObject(pUCB_class, pArguments);
    printf("   'pUCB' = '%s'\n", PyBytes_AsString(PyObject_Str(pUCB)));

    return loop(pUCB, pMAB, horizon);

    // pName = PyBytes_FromString(argv[1]);
    // printf("Looking for module = '%s'\n", argv[1]);

    // pModule = PyImport_Import(pName);
    // if (pModule != NULL) {
    //     pArguments = PyTuple_New(1);

    //     pValue = PyLong_FromString(argv[3], NULL, 10);
    //     if (pValue == NULL) {
    //         return 1;
    //     }
    //     PyTuple_SetItem(pArguments, 0, pValue);

    //     pValue = PyLong_FromString(argv[4], NULL, 10);
    //     if (pValue == NULL) {
    //         return 1;
    //     }
    //     PyTuple_SetItem(pArguments, 1, pValue);

    //     pFunc = PyObject_GetAttrString(pModule, argv[2]);
    //     if (pFunc && PyCallable_Check(pFunc)) {
    //         printf("Calling function function = '%s' with ('%s', '%s')\n", argv[2], argv[3], argv[4]);
    //         // printf("Calling function function = '%s' with ('%s',)\n", argv[2], argv[3]);
    //         pValue = PyObject_CallObject(pFunc, pArguments);
    //         if (pValue != NULL) {
    //             printf("Value returned from the function = '%s'\n", PyBytes_AsString(PyObject_Str(pValue)));
    //         } else {
    //             PyErr_Print();
    //         }
    //     } else {
    //         if (PyErr_Occurred())
    //             PyErr_Print();
    //         fprintf(stderr, "Cannot find function '%s'\n", argv[2]);
    //     }
    // }
    // else {
    //     PyErr_Print();
    //     fprintf(stderr, "Failed to load '%s'\n", argv[1]);
    //     return 1;
    // }

    // return 0;
}
