#ifndef _PERCEPTRON_H_
#define _PERCEPTRON_H_

#include "matrix.h"
#include <time.h>

typedef struct struct_Perceptron Perceptron;

int allocatePerceptron(Perceptron **perc, size_t weightNumber);

int setPerceptronWeight(Perceptron *perc, size_t weightNumber, double value);
int getPerceptronWeight(const Perceptron *perc, size_t weightNumber, double *value);

void setPerceptronBias(Perceptron *perc, double value);
double getPerceptronBias(const Perceptron* perc);

void generatePerceptron(Perceptron *perc, time_t seed);

double sigmoid(double x);
int forward(const Perceptron *perc, const Comatrix *x, Matrix *res);
int computeLoss(const Matrix *yTrue, const Matrix *yPred, double *res);
int fit(Perceptron *perc, Comatrix *xTrain, Matrix *yTrain, Comatrix *xVal, Matrix *yVal,
            size_t epochs, double lr, size_t batchSize, double *lossesTrain, double *lossesVal);

int predict(const Perceptron *perc, const Comatrix *x, int *res);

#endif //_PERCEPTRON_H_