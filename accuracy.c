#include "accuracy.h"
#include <math.h>

int countAccuracy(Perceptron *perc, Matrix **matArray, double *trainAccuracy, double *valAccuracy,
            int *trainCorrectnesses, int *valCorrectnesses){

    if(getPerceptronWeightNumber(perc) != getMatrixRowNumber(matArray[0])) return 0;
    if(getPerceptronWeightNumber(perc) != getMatrixRowNumber(matArray[2])) return 0;
    if(getMatrixColumnNumber(matArray[0]) != getMatrixColumnNumber(matArray[1])) return 0;
    if(getMatrixColumnNumber(matArray[2]) != getMatrixColumnNumber(matArray[3])) return 0;
    if(getMatrixRowNumber(matArray[1]) != 1) return 0;
    if(getMatrixRowNumber(matArray[3]) != 1) return 0;

    size_t weightNumber = getPerceptronWeightNumber(perc);
    size_t trainSampleNumber = getMatrixColumnNumber(matArray[0]);
    size_t valSampleNumber = getMatrixColumnNumber(matArray[2]);

    size_t trueNumber = 0;

    Matrix *testColumn;
    if(!allocateMatrix(&testColumn, weightNumber, 1)) return -1;

    double lfSubserv;
    int intSubserv;
    int correctness;
    size_t sample, feature;

    for(sample = 0; sample < trainSampleNumber; ++sample){

        for(feature = 0; feature < weightNumber; ++feature){
            getMatrixElement(matArray[0], feature, sample, &lfSubserv);
            setMatrixElement(testColumn, feature, 0, lfSubserv);
        }

        predict(perc, testColumn, &intSubserv);
        getMatrixElement(matArray[1], 0, sample, &lfSubserv);

        correctness = ((double)intSubserv == lfSubserv);
        if(correctness) ++trueNumber;
        if(trainCorrectnesses) trainCorrectnesses[sample] = correctness;

    }

    *trainAccuracy = (double)trueNumber/trainSampleNumber;

    trueNumber = 0;

    for(sample = 0; sample < valSampleNumber; ++sample){

        for(feature = 0; feature < weightNumber; ++feature){
            getMatrixElement(matArray[2], feature, sample, &lfSubserv);
            setMatrixElement(testColumn, feature, 0, lfSubserv);
        }

        predict(perc, testColumn, &intSubserv);
        getMatrixElement(matArray[3], 0, sample, &lfSubserv);

        correctness = ((double)intSubserv == lfSubserv);
        if(correctness) ++trueNumber;
        if(valCorrectnesses) valCorrectnesses[sample] = correctness;

    }

    *valAccuracy = (double)trueNumber/valSampleNumber;

    freeMatrix(testColumn);

    return 1;

}