#include "third_experiment.h"
#include <math.h>
#include <stdio.h>

static const size_t epochs = 100;
static const double lr = 0.1;
static const size_t batchSize = 32;

double normalDistribution(double mu, double sigmaSquare){

    static double spare;
    static int hasSpare = 0;

    if(hasSpare){
        hasSpare = 0;
        return spare*sqrt(sigmaSquare) + mu;
    }

    double u, v, s;
    do{
        u = (double)rand()/RAND_MAX * 2 - 1;
        v = (double)rand()/RAND_MAX * 2 - 1;
        s = u*u + v*v;
    }while(s == 0 || s >= 1);
    
    s = sqrt(-2*log(s)/s);

    spare = v*s;
    hasSpare = 1;

    return u*s*sqrt(sigmaSquare) + mu;

}

int thirdExperiment(Matrix **matArray){

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

    for(size_t weight = 0; weight < weightNumber; ++weight){
        setPerceptronWeight(perc, weight, 0);
    }
    setPerceptronBias(perc, 0);

    fit(perc, matArray[0], matArray[1], matArray[2], matArray[3], epochs, lr, batchSize, lossesTrain, lossesVal);
    countAccuracy(perc, matArray, &trainAccuracy, &valAccuracy, NULL, NULL);

    printf("The initialization is null.\n");

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
    fit(perc, matArray[0], matArray[1], matArray[2], matArray[3], epochs, lr, batchSize, lossesTrain, lossesVal);
    countAccuracy(perc, matArray, &trainAccuracy, &valAccuracy, NULL, NULL);

    printf("The initialization is default.\n");

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

    srand(time(NULL));

    for(size_t weight = 0; weight < weightNumber; ++weight){
        setPerceptronWeight(perc, weight, normalDistribution(0, 10));
    }
    setPerceptronBias(perc, 0);

    fit(perc, matArray[0], matArray[1], matArray[2], matArray[3], epochs, lr, batchSize, lossesTrain, lossesVal);
    countAccuracy(perc, matArray, &trainAccuracy, &valAccuracy, NULL, NULL);

    printf("The initialization is N(0, 10).\n");

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