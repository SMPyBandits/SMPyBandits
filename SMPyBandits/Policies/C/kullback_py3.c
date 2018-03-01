/*
-*- coding: utf-8 -*-
authors: Olivier Cappé and Aurélien Garivier

C version of some Kullback-Leibler utilities provided in file "kullback.py"
*/

#include <Python.h>
#include <math.h>
// #defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define eps 1e-15
#define inf 1e300
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) < (Y)) ? (Y) : (X))

double _klBern(double p, double q){
	p = p<1-eps? (p>eps ? p:eps) : 1-eps;
	q = q<1-eps? (q>eps ? q:eps) : 1-eps;
	return p*log(p/q) + (1-p)*log((1-p)/(1-q));
}

static PyObject* klBern(PyObject* self, PyObject* args)
{
    //const char *command;
    double x,y;

    if (!PyArg_ParseTuple(args, "dd", &x, &y))
        return NULL;

    return Py_BuildValue("d", _klBern(x,y));
}

double _klBin(double p, double q, int n){
    p = p<1-eps? (p>eps ? p:eps) : 1-eps;
    q = q<1-eps? (q>eps ? q:eps) : 1-eps;
    return n * (p*log(p/q) + (1-p)*log((1-p)/(1-q)) );
}

static PyObject* klBin(PyObject* self, PyObject* args)
{
    //const char *command;
    double x,y;
    int n;

    if (!PyArg_ParseTuple(args, "ddi", &x, &y, &n))
        return NULL;

    return Py_BuildValue("d", _klBin(x,y,n));
}

double _klPoisson(double x, double y){
	x = x>eps ? x:eps;
	y = y>eps ? y:eps;
	return y-x+x*log(x/y);
}

static PyObject* klPoisson(PyObject* self, PyObject* args)
{
    //const char *command;
    double x,y;

    if (!PyArg_ParseTuple(args, "dd", &x, &y))
        return NULL;

    return Py_BuildValue("d", _klPoisson(x,y));
}

double _klExp(double x, double y){
	x = x>eps ? x:eps;
	y = y>eps ? y:eps;
	return x/y - 1 - log(x/y);
}

static PyObject* klExp(PyObject* self, PyObject* args)
{
    //const char *command;
    double x,y;

    if (!PyArg_ParseTuple(args, "dd", &x, &y))
        return NULL;

    return Py_BuildValue("d", _klExp(x,y));
}

double _klGamma(double x, double y, double a){
    x = x>eps ? x:eps;
    y = y>eps ? y:eps;
    return a * (x/y - 1 - log(x/y));
}

static PyObject* klGamma(PyObject* self, PyObject* args)
{
    //const char *command;
    double x,y,a;

    if (!PyArg_ParseTuple(args, "ddd", &x, &y, &a))
        return NULL;

    return Py_BuildValue("d", _klGamma(x,y,a));
}

double _klGauss(double x, double y, double sig2){
	return (x-y)*(x-y)/(2*sig2);
}

static PyObject* klGauss(PyObject* self, PyObject* args)
{
    //const char *command;
    double x,y,sig2;

    if (!PyArg_ParseTuple(args, "ddd", &x, &y, &sig2))
        return NULL;

    return Py_BuildValue("d", _klGauss(x,y,sig2));
}

double _klucb(double x, double d, double (*div)(double, double), double l, double u, double precision){
  while (u-l>precision){
		double m = (l+u)/2;
		if ((*div)(x, m)>d) u = m; else l = m;
	}
  return (l+u)/2;
}

/*
static PyObject* klucb(PyObject* self, PyObject* args) // PAS BON !
{
    //const char *command;
    double x,d;
    double precision;
    if (!PyArg_ParseTuple(args, "ddd", &x, &d, &precision))
        return NULL;

    return Py_BuildValue("d", _klucb(x,d,div,upperBound,precision));
}
*/

double _klucbGauss(double x, double d, double sig2){
	return x + sqrt(2 * sig2 * d);
}

static PyObject* klucbGauss(PyObject* self, PyObject* args)
{
    //const char *command;
    double x,d,sig2;

    if (!PyArg_ParseTuple(args, "ddd", &x, &d, &sig2))
        return NULL;

    return Py_BuildValue("d", _klucbGauss(x,d,sig2));
}

double _klucbPoisson(double x, double d, double precision){
	// lowerbound du tcl?
	double upperbound = x+d+sqrt(d*d+2*x*d);
	return _klucb(x, d, _klPoisson, x, upperbound, precision);
}

static PyObject* klucbPoisson(PyObject* self, PyObject* args)
{
    //const char *command;
    double x,d,precision;

    if (!PyArg_ParseTuple(args, "ddd", &x, &d, &precision))
        return NULL;

    return Py_BuildValue("d", _klucbPoisson(x,d,precision));
}

double _klucbBern(double x, double d, double precision){
	// lowerbound du tcl?
	double upperbound = _klucbGauss(x,d,1.);
	upperbound = (upperbound<1.)?upperbound:1.;
	return _klucb(x, d, _klBern, x, upperbound, precision);
}

static PyObject* klucbBern(PyObject* self, PyObject* args)
{
    //const char *command;
    double x,d,precision;

    if (!PyArg_ParseTuple(args, "ddd", &x, &d, &precision))
        return NULL;

    return Py_BuildValue("d", _klucbBern(x,d,precision));
}

double _klucbExp(double x, double d, double precision){
	double lowerbound =  d<1.61?x*exp(d):x/(1+d-sqrt(d*d+2*d));
	double upperbound =  d<0.77?x/(1+2./3*d-sqrt(4./9*d*d+2*d)) : x*exp(d+1); // safe, klexp(x,y) >= e^2/(2*(1-2e/3)) if x=y(1-e)
	return _klucb(x, d, _klExp, lowerbound, upperbound, precision);
}

static PyObject* klucbExp(PyObject* self, PyObject* args)
{
    //const char *command;
    double x,d,precision;

    if (!PyArg_ParseTuple(args, "ddd", &x, &d, &precision))
        return NULL;

    return Py_BuildValue("d", _klucbExp(x,d,precision));
}

// FIXME this one is wrong!
double _klucbGamma(double x, double d, double precision){
    double lowerbound =  d<1.61?x*exp(d):x/(1+d-sqrt(d*d+2*d));
    double upperbound =  d<0.77?x/(1+2./3*d-sqrt(4./9*d*d+2*d)) : x*exp(d+1); // safe, klexp(x,y) >= e^2/(2*(1-2e/3)) if x=y(1-e)
    return _klucb(x, d, _klGamma, MIN(lowerbound, -100), MAX(upperbound, 100), precision);
}

// FIXME this one is wrong!
static PyObject* klucbGamma(PyObject* self, PyObject* args)
{
    //const char *command;
    double x,d,precision;

    if (!PyArg_ParseTuple(args, "ddd", &x, &d, &precision))
        return NULL;

    return Py_BuildValue("d", _klucbGamma(x,d,precision));
}

double _reseqp(int size, double* p, double* V, double klMax, double mV){
		int i;
		double u=0, y=0, yp=0;
		double l = mV + 0.1;
		double tol = 1e-4;
		//if (mV<min(V)+tol) return inf;
		for (i=0; i<size; i++) if (p[i]>0){
				u += p[i] / (l - V[i]);
				y += p[i] * log(l-V[i]);
		}
		y += log(u) - klMax; //printf("p[0] = %g, V[0] = %g, u = %g, y = %g, tol = %g\n", p[0], V[0], u, fabs(y), tol);
		while (fabs(y)>tol){
			yp = 0;
			for (i=0; i<size; i++) yp += p[i]/((l-V[i])*(l-V[i]));
			yp = u - yp / u;
			l = l - y / yp;
			if (l<mV) l = (l + y/yp +mV)/2;
			u = 0; y = 0;
			for (i=0; i<size; i++) if (p[i]>0){
					u += p[i] / (l - V[i]);
					y += p[i] * log(l-V[i]);
			}
			y += log(u) - klMax; // printf("y = %g\n", y);
		}
		return(l);
}

void _maxEV(int size, double* p, double* V, double klMax, double* Uq){
	int i, j=0;
	double y=0, u=0, rb;
	double M[2] = {-inf, -inf}; // maximum of not loaded, and loaded values
	double m = inf; // minimum of loaded values
	for (i=0; i<size; i++){
		Uq[i] = 0;
		if (V[i]>M[p[i]>0]){
			M[p[i]>0] = V[i];
			if(p[i]==0) j = i; // store index of maximum with p[i]==0
		}
		if ((V[i]<m) && (p[i]>0)) m = V[i];
	}
	if (M[0]>M[1]){
		for (i=0; i<size; i++) if (p[i]>0){
			u += p[i] / (M[0] - V[i]);
			y += p[i] * log(M[0] - V[i]);
		}
//		printf("eta = %g, y = %g\n", M[0], y+log(u));
		if ((y += log(u) - klMax)<0){
			rb = exp(y);
			u = 0;
			for (i=0; i<size; i++) if (p[i]>0) u+= (Uq[i] = p[i]/(M[0]-V[i]));
			for (i=0; i<size; i++) if (p[i]>0) Uq[i] *= rb/u;
			Uq[j] = 1-rb;
			return;
		}
	}
	if (M[1]-m<1e-8) // tol !!!
		for(i =0; i<size; i++) if (p[i]>0) {Uq[i]=1; return;}
	y = _reseqp(size, p, V, klMax, M[1]);
//	printf("eta = %g\n", y);
	u = 0;
	for (i=0; i<size; i++) if (p[i]>0) u+= (Uq[i] = p[i]/(y-V[i]));
	for (i=0; i<size; i++) if (p[i]>0) Uq[i] /= u;
}

static PyObject* maxEV(PyObject* self, PyObject* args)
{
    //const char *command;
    PyObject *arg1=NULL, *arg2=NULL;
    PyObject *arr1=NULL, *arr2=NULL;
    double klMax;

    if (!PyArg_ParseTuple(args, "OOd", &arg1, &arg2, &klMax))
        return NULL;

	PyObject * out = PyArray_SimpleNew(1, PyArray_DIMS(arg1), PyArray_DOUBLE);
	//uncomment for safety: it's type and contigency checking
    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);

    double *p, *q, *res;
    int size = PyArray_DIMS(arg1)[0];
    p = (double *)PyArray_DATA(arr1); //uncomment for safety: it's type and contigency checking,
    q = (double *)PyArray_DATA(arr2);
/*    p = (double *)PyArray_DATA(arg1); // the time won is miserable
    q = (double *)PyArray_DATA(arg2);
*/
    res = (double *)PyArray_DATA(out);

    _maxEV(size,p,q,klMax,res);

    Py_DECREF(arr1);
    Py_DECREF(arr2);
 //   Py_INCREF(out);
    return out;
}



static PyMethodDef kullbackMethods[] = {
    {"klBern", klBern, METH_VARARGS, "klBern(x, y): Calculate the binary Kullback-Leibler divergence."},
    {"klBin", klBin, METH_VARARGS, "klBin(x, y, n): Calculate the Kullback-Leibler divergence for Binomial distributions of same n."},
    {"klPoisson", klPoisson, METH_VARARGS, "klPoisson(x, y): Calculate the Kullback-Leibler divergence for Poisson distributions."},
    {"klExp", klExp, METH_VARARGS, "klExp(x, y): Calculate the Kullback-Leibler for Exponential distributions."},
    {"klGamma", klGamma, METH_VARARGS, "klGamma(x, y, a=1): Calculate the Kullback-Leibler for Gamma distributions."},
    {"klGauss", klGauss, METH_VARARGS, "klGauss(x, y, sig2): Calculate the Kullback-Leibler for Gaussian distributions."},
    {"klucbGauss", klucbGauss, METH_VARARGS, "klucbGauss(x, d, sig2, precision=0.): UCB for Gaussian observations."},
    {"klucbPoisson", klucbPoisson, METH_VARARGS, "klucbPoisson(x, d, precision=1e-6): UCB for Poisson observations."},
    {"klucbBern", klucbBern, METH_VARARGS, "klucbBern(x, d, precision=1e-6): UCB for Bernoulli observations."},
    {"klucbExp", klucbExp, METH_VARARGS, "klucbExp(x, d, precision=1e-6): UCB for Exponential observations."},
    {"klucbGamma", klucbGamma, METH_VARARGS, "klucbGamma(x, d, precision=1e-6): UCB for Gamma observations."},
    {"maxEV", maxEV, METH_VARARGS, "maxEV(p, V, klMax): maximize linear function under KL constraint."},
  //{"klucb", klucb, METH_VARARGS, "Compute the kl-ucb at x with distance d and precision prec."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef kullbackModuleDef =
{
    PyModuleDef_HEAD_INIT,
    "kullback",  /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    kullbackMethods
};

PyMODINIT_FUNC PyInit_kullback(void) {
    // This is for Python 3. Cf. http://stackoverflow.com/a/28306354/5889533
    return PyModule_Create(&kullbackModuleDef);
};
