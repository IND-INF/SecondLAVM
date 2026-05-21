#include "first_experiment.h"
#include <stdio.h>

static const size_t epochs = 100;
static const size_t batchSize = 32;

int firstExperiment(Matrix **matArray){

    if(getMatrixRowNumber(matArray[0]) != getMatrixRowNumber(matArray[2])) return 0;
    if(getMatrixColumnNumber(matArray[0]) != getMatrixColumnNumber(matArray[1])) return 0;
    if(getMatrixColumnNumber(matArray[2]) != getMatrixColumnNumber(matArray[3])) return 0;
    if(getMatrixRowNumber(matArray[1]) != 1) return 0;
    if(getMatrixRowNumber(matArray[3]) != 1) return 0;

    size_t weightNumber = getMatrixRowNumber(matArray[0]);
    size_t currentEpoch;

    double lossesTrain[epochs];
    double lossesVal[epochs];
    
    double trainAccuracy;
    double valAccuracy;

    Perceptron *perc;
    allocatePerceptron(&perc, weightNumber);

    generatePerceptron(perc, time(NULL));
    fit(perc, matArray[0], matArray[1], matArray[2], matArray[3], epochs, 0.001, batchSize, lossesTrain, lossesVal);
    countAccuracy(perc, matArray, &trainAccuracy, &valAccuracy, NULL, NULL);

    printf("The learning rate is 0.001.\n");

    printf("The train losses:\n");
    for(currentEpoch = 0; currentEpoch < epochs; ++currentEpoch){
        printf("%.16f\n", lossesTrain[currentEpoch]);
    }

    printf("The validation losses:\n");
    for(currentEpoch = 0; currentEpoch < epochs; ++currentEpoch){
        printf("%.16f\n", lossesVal[currentEpoch]);
    }

    printf("The train accuracy is %.16f.\n", trainAccuracy);
    printf("The validation accuracy is %.16f.\n", valAccuracy);

    generatePerceptron(perc, time(NULL));
    fit(perc, matArray[0], matArray[1], matArray[2], matArray[3], epochs, 0.01, batchSize, lossesTrain, lossesVal);
    countAccuracy(perc, matArray, &trainAccuracy, &valAccuracy, NULL, NULL);

    printf("The learning rate is 0.01.\n");

    printf("The train losses:\n");
    for(currentEpoch = 0; currentEpoch < epochs; ++currentEpoch){
        printf("%.16f\n", lossesTrain[currentEpoch]);
    }

    printf("The validation losses:\n");
    for(currentEpoch = 0; currentEpoch < epochs; ++currentEpoch){
        printf("%.16f\n", lossesVal[currentEpoch]);
    }

    printf("The train accuracy is %.16f.\n", trainAccuracy);
    printf("The validation accuracy is %.16f.\n", valAccuracy);
    
    generatePerceptron(perc, time(NULL));
    fit(perc, matArray[0], matArray[1], matArray[2], matArray[3], epochs, 0.5, batchSize, lossesTrain, lossesVal);
    countAccuracy(perc, matArray, &trainAccuracy, &valAccuracy, NULL, NULL);

    printf("The learning rate is 0.5.\n");

    printf("The train losses:\n");
    for(currentEpoch = 0; currentEpoch < epochs; ++currentEpoch){
        printf("%.16f\n", lossesTrain[currentEpoch]);
    }

    printf("The validation losses:\n");
    for(currentEpoch = 0; currentEpoch < epochs; ++currentEpoch){
        printf("%.16f\n", lossesVal[currentEpoch]);
    }

    printf("The train accuracy is %.16f.\n", trainAccuracy);
    printf("The validation accuracy is %.16f.\n", valAccuracy);

    generatePerceptron(perc, time(NULL));
    fit(perc, matArray[0], matArray[1], matArray[2], matArray[3], epochs, 1, batchSize, lossesTrain, lossesVal);
    countAccuracy(perc, matArray, &trainAccuracy, &valAccuracy, NULL, NULL);

    printf("The learning rate is 1.\n");

    printf("The train losses:\n");
    for(currentEpoch = 0; currentEpoch < epochs; ++currentEpoch){
        printf("%.16f\n", lossesTrain[currentEpoch]);
    }

    printf("The validation losses:\n");
    for(currentEpoch = 0; currentEpoch < epochs; ++currentEpoch){
        printf("%.16f\n", lossesVal[currentEpoch]);
    }

    printf("The train accuracy is %.16f.\n", trainAccuracy);
    printf("The validation accuracy is %.16f.\n", valAccuracy);

    freePerceptron(perc);

    return 0;
    
}