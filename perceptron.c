#include "perceptron.h"
#include <math.h>

typedef struct struct_Perceptron{
    Matrix *w;
    double b;
} Perceptron;

int allocatePerceptron(Perceptron **perc, size_t weightNumber){

    if(!(*perc = malloc(sizeof(Perceptron)))) return 0;

    if(!allocateMatrix(&((*perc)->w), 1, weightNumber)) return 0;

    return 1;

}

void freePerceptron(Perceptron *perc){
    freeMatrix(perc->w);
    free(perc);
}

size_t getPerceptronWeightNumber(const Perceptron *perc){
    return getMatrixColumnNumber(perc->w);
}

int setPerceptronWeight(Perceptron *perc, size_t weightNumber, double value){
    return setMatrixElement(perc->w, 0, weightNumber, value);
}

int getPerceptronWeight(const Perceptron *perc, size_t weightNumber, double *value){
    return getMatrixElement(perc->w, 0, weightNumber, value);
}

void setPerceptronBias(Perceptron *perc, double value){
    perc->b = value;
}

double getPerceptronBias(const Perceptron* perc){
    return perc->b;
}

void generatePerceptron(Perceptron *perc, time_t seed){

    srand(seed);

    size_t weightNumber = getMatrixColumnNumber(perc->w);
    for(size_t number = 0; number < weightNumber; ++number){
        setMatrixElement(perc->w, 0, number, ((double)rand())/RAND_MAX);
    }

    perc->b = 0;

}

double sigmoid(double x){
    return 1.0/(1.0 + exp(-x));
}

int forward(const Perceptron *perc, const Matrix *x, Matrix *res){

    if(!multiplyMatrices(perc->w, x, res)) return 0;

    size_t ansNumber = getMatrixColumnNumber(res);
    double element;
    for(size_t number = 0; number < ansNumber; ++number){

        getMatrixElement(res, 0, number, &element);

        element += perc->b;
        element = sigmoid(element);

        setMatrixElement(res, 0, number, element);

    }

    return 1;

}

int computeLoss(const Matrix *yTrue, const Matrix *yPred, double *res){

    if(getMatrixColumnNumber(yTrue) != getMatrixColumnNumber(yPred)) return 0;
    if(getMatrixRowNumber(yTrue) != 1) return 0;
    if(getMatrixRowNumber(yPred) != 1) return 0;

    *res = 0;

    size_t ansNumber = getMatrixColumnNumber(yTrue);
    double trueElement, predElement;
    for(size_t number = 0; number < ansNumber; number++){

        getMatrixElement(yTrue, 0, number, &trueElement);
        getMatrixElement(yPred, 0, number, &predElement);

        *res -= trueElement*log(predElement) + (1-trueElement)*log(1-predElement);

    }

    *res /= ansNumber;

    return 1;

}

void shuffleColumns(size_t *columns, size_t columnNumber){

    int randRes;
    size_t sizeSubserv;

    for(size_t number = columnNumber; number > 0; number--){
            
        randRes = rand();
        while(randRes >= (RAND_MAX + 1)/number * number){
            randRes = rand();
        }

        sizeSubserv = columns[number-1];
        columns[number-1] = columns[randRes%number];
        columns[randRes%number] = sizeSubserv;

    }

}

int fit(Perceptron *perc, Matrix *xTrain, Matrix *yTrain, Matrix *xVal, Matrix *yVal,
            size_t epochs, double lr, size_t batchSize, double *lossesTrain, double *lossesVal){
    
    if(getMatrixRowNumber(xTrain) != getMatrixColumnNumber(perc->w)) return 0;
    if(getMatrixRowNumber(xVal) != getMatrixColumnNumber(perc->w)) return 0;
    if(getMatrixColumnNumber(xTrain) != getMatrixColumnNumber(yTrain)) return 0;
    if(getMatrixColumnNumber(xVal) != getMatrixColumnNumber(yVal)) return 0;
    if(getMatrixColumnNumber(xTrain) > RAND_MAX + 1) return 0;
    if(getMatrixColumnNumber(xVal) > RAND_MAX + 1) return 0;
    if(batchSize < 1) return 0;
    
    time_t seed = time(NULL);
    srand(seed);

    size_t rowNumber = getMatrixRowNumber(xTrain);
    size_t trainColumnNumber = getMatrixColumnNumber(xTrain);
    size_t valColumnNumber = getMatrixColumnNumber(xVal);
    size_t trainBatchNumber = trainColumnNumber/batchSize;
    size_t valBatchNumber = valColumnNumber/batchSize;

    Matrix *xBatch;
    if(!allocateMatrix(&xBatch, rowNumber, batchSize)) return -1;

    Matrix *xBatchTranspose;
    if(!allocateMatrix(&xBatchTranspose, batchSize, rowNumber)){
        freeMatrix(xBatch);
        return -1;
    }

    Matrix *yBatch;
    if(!allocateMatrix(&yBatch, 1, batchSize)){
        freeMatrix(xBatchTranspose);
        freeMatrix(xBatch);
        return -1;
    }

    Matrix *yPred;
    if(!allocateMatrix(&yPred, 1, batchSize)){
        freeMatrix(yBatch);
        freeMatrix(xBatchTranspose);
        freeMatrix(xBatch);
        return -1;
    }

    Matrix *lossDerRespW;
    if(!allocateMatrix(&lossDerRespW, 1, rowNumber)){
        freeMatrix(yPred);
        freeMatrix(yBatch);
        freeMatrix(xBatchTranspose);
        freeMatrix(xBatch);
        return -1;
    }
    
    size_t columnsTrain[trainColumnNumber];
    for(size_t number = 0; number < trainColumnNumber; number++){
        columnsTrain[number] = number;
    }

    double loss, totalLoss;
    double lossDerRespB;

    double lfSubserv;
    for(size_t epochNumber = 0, batch, batchStart, batchColumn, batchRow; epochNumber < epochs; ++epochNumber){

        shuffleColumns(columnsTrain, trainColumnNumber);

        totalLoss = 0;
        for(batch = 0; batch < trainBatchNumber; ++batch){

            batchStart = batch * batchSize;

            for(batchColumn = 0; batchColumn < batchSize; ++batchColumn){

                for(batchRow = 0; batchRow < rowNumber; ++batchRow){
                    getMatrixElement(xTrain, batchRow, columnsTrain[batchStart + batchColumn], &lfSubserv);
                    setMatrixElement(xBatch, batchRow, batchColumn, lfSubserv);
                }

                getMatrixElement(yTrain, 0, columnsTrain[batchStart + batchColumn], &lfSubserv);
                setMatrixElement(yBatch, 0, batchColumn, lfSubserv);
                
            }

            forward(perc, xBatch, yPred);

            computeLoss(yBatch, yPred, &loss);
            totalLoss += loss;

            subtractAndAssignMatrix(yPred, yBatch);
            matrixTranspose(xBatch, xBatchTranspose);

            multiplyMatrices(yPred, xBatchTranspose, lossDerRespW);
            scalarlyMultiplyAndAssignMatrix(lr/(double)batchSize, lossDerRespW);

            lossDerRespB = 0;
            for(batchColumn = 0; batchColumn < batchSize; ++batchColumn){
                getMatrixElement(yPred, 0, batchColumn, &lfSubserv);
                lossDerRespB += lfSubserv;
            }
            lossDerRespB *= lr/(double)batchSize;

            subtractAndAssignMatrix(perc->w, lossDerRespW);
            perc->b -= lossDerRespB;
            
        }
        lossesTrain[epochNumber] = totalLoss/trainBatchNumber;

        totalLoss = 0;
        for(batch = 0; batch < valBatchNumber; ++batch){

            batchStart = batch * batchSize;

            for(batchColumn = 0; batchColumn < batchSize; ++batchColumn){

                for(batchRow = 0; batchRow < rowNumber; ++batchRow){
                    getMatrixElement(xVal, batchRow, batchStart + batchColumn, &lfSubserv);
                    setMatrixElement(xBatch, batchRow, batchColumn, lfSubserv);
                }

                getMatrixElement(yVal, 0, batchStart + batchColumn, &lfSubserv);
                setMatrixElement(yBatch, 0, batchColumn, lfSubserv);
                
            }

            forward(perc, xBatch, yPred);

            computeLoss(yBatch, yPred, &loss);
            totalLoss += loss;
            
        }
        if(valBatchNumber) lossesVal[epochNumber] = totalLoss/valBatchNumber;

    }

    freeMatrix(xBatch);
    freeMatrix(xBatchTranspose);
    freeMatrix(yBatch);
    freeMatrix(yPred);
    freeMatrix(lossDerRespW);

    return 1;

}

int predict(const Perceptron *perc, const Matrix *x, int *res){

    Matrix *ans;
    allocateMatrix(&ans, 1, 1);

    if(!multiplyMatrices(perc->w, x, ans)) return 0;

    double preRes;
    getMatrixElement(ans, 0, 0, &preRes);

    if(preRes < .5) *res = 0;
    else *res = 1;

    return 1;

}