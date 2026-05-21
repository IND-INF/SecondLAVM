#include "second_experiment.h"
#include <stdio.h>

static const size_t epochs = 100;
static const double lr = 0.1;

int secondExperiment(Matrix **matArray){

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
    fit(perc, matArray[0], matArray[1], matArray[2], matArray[3], epochs, lr, 1, lossesTrain, lossesVal);
    countAccuracy(perc, matArray, &trainAccuracy, &valAccuracy, NULL, NULL);

    printf("The batch size is 1.\n");

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
    fit(perc, matArray[0], matArray[1], matArray[2], matArray[3], epochs, lr, 16, lossesTrain, lossesVal);
    countAccuracy(perc, matArray, &trainAccuracy, &valAccuracy, NULL, NULL);

    printf("The batch size is 16.\n");

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
    fit(perc, matArray[0], matArray[1], matArray[2], matArray[3], epochs, lr, 64, lossesTrain, lossesVal);
    countAccuracy(perc, matArray, &trainAccuracy, &valAccuracy, NULL, NULL);

    printf("The batch size is 64.\n");

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
    fit(perc, matArray[0], matArray[1], matArray[2], matArray[3], epochs, lr, 256, lossesTrain, lossesVal);
    countAccuracy(perc, matArray, &trainAccuracy, &valAccuracy, NULL, NULL);

    printf("The batch size is 256.\n");

    printf("The train losses:\n");
    for(currentEpoch = 0; currentEpoch < epochs; ++currentEpoch){
        printf("%.16f\n", lossesTrain[currentEpoch]);
    }

    printf("The train accuracy is %.16f.\n", trainAccuracy);

    freePerceptron(perc);

    return 0;
    
}