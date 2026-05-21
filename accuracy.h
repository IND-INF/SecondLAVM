#ifndef _ACCURACY_H_
#define _ACCURACY_H_

#include "perceptron.h"

int countAccuracy(Perceptron *perc, Matrix **matArray, double *trainAccuracy, double *valAccuracy,
            int *trainCorrectnesses, int *valCorrectnesses);

#endif //_ACCURACY_H_