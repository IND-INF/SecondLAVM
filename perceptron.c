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
        setMatrixElement(perc->w, 0, number, 4*((double)rand())/RAND_MAX);
    }

    perc->b = 0;

}

double sigmoid(double x){
    return 1/(1 + exp(-x));
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

int fit(Perceptron *perc, Matrix *xTrain, Matrix *yTrain, Matrix *xVal, Matrix *yVal,
            size_t epochs, double lr, size_t batchSize, double *lossesTrain, double *lossVal){
    
    if(getMatrixRowNumber(xTrain) != getMatrixColumnNumber(perc->w)) return 0;
    if(getMatrixRowNumber(xVal) != getMatrixColumnNumber(perc->w)) return 0;
    if(getMatrixColumnNumber(xTrain) != getMatrixColumnNumber(yTrain)) return 0;
    if(getMatrixColumnNumber(xVal) != getMatrixColumnNumber(yVal)) return 0;
    if(getMatrixColumnNumber(xTrain) > RAND_MAX + 1) return 0;
    
    time_t seed = time(NULL);
    srand(seed);

    size_t rowNumber = getMatrixRowNumber(xTrain);
    size_t columnNumber = getMatrixColumnNumber(xTrain);
    size_t batchNumber = columnNumber/batchSize;
    size_t valColumnNumber = getMatrixColumnNumber(xVal);

    Matrix *xBatch;
    if(!allocateMatrix(&xBatch, rowNumber, batchSize)) return -1;

    Matrix *xBatchTranspose;
    if(!allocateMatrix(&xBatchTranspose, batchSize, rowNumber)) return -1;

    Matrix *yBatch;
    if(!allocateMatrix(&yBatch, 1, batchSize)) return -1;

    Matrix *yPred;
    if(!allocateMatrix(&yPred, 1, batchSize)) return -1;

    Matrix *yValPred;
    if(!allocateMatrix(&yValPred, 1, valColumnNumber)) return -1;

    Matrix *lossDerRespW;
    if(!allocateMatrix(&lossDerRespW, 1, rowNumber)) return -1;
    
    size_t columns[columnNumber];
    for(size_t number = 0; number < columnNumber; number++){
        columns[number] = number;
    }

    double loss;
    double lossDerRespB;

    double lfSubserv;
    size_t sizeSubserv;
    for(size_t epochNumber = 0, batchEnd, batchCurrent, batchRow; epochNumber < epochs; ++epochNumber){

        int randRes;
        for(size_t number = columnNumber; number > 0; number--){
            
            randRes = rand();
            while(randRes >= (RAND_MAX + 1)/number * number){
                randRes = rand();
            }

            sizeSubserv = columns[number-1];
            columns[number-1] = columns[randRes%number];
            columns[randRes%number] = sizeSubserv;

        }

        for(batchEnd = batchSize, batchCurrent = 0; batchEnd <= columnNumber; batchEnd += batchSize){

            for(batchCurrent = batchEnd - batchSize; batchCurrent < batchEnd; ++batchCurrent){

                for(batchRow = 0; batchRow < rowNumber; ++batchRow){
                    getMatrixElement(xTrain, batchRow, columns[batchCurrent], &lfSubserv);
                    setMatrixElement(xBatch, batchRow, batchCurrent, lfSubserv);
                }

                getMatrixElement(yTrain, 0, columns[batchCurrent], &lfSubserv);
                setMatrixElement(yBatch, 0, batchCurrent, lfSubserv);
                
            }

            forward(perc, xBatch, yPred);

            computeLoss(yBatch, yPred, &loss);
            lossesTrain[epochNumber*batchNumber + batchEnd/batchSize - 1] = loss;

            subtractAndAssignMatrix(yPred, yBatch);
            matrixTranspose(xBatch, xBatchTranspose);

            multiplyMatrices(yPred, xBatchTranspose, lossDerRespW);
            scalarlyMultiplyAndAssignMatrix(lr/rowNumber, lossDerRespW);

            lossDerRespB = 0;
            for(batchCurrent = batchEnd - batchSize; batchCurrent < batchEnd; ++batchCurrent){
                getMatrixElement(yPred, 0, batchCurrent, &lfSubserv);
                lossDerRespB += lfSubserv;
            }
            lossDerRespB *= lr/rowNumber;

            subtractAndAssignMatrix(perc->w, lossDerRespW);
            perc->b -= lossDerRespB;
            
        }
    }

    forward(perc, xVal, yValPred);
    computeLoss(yVal, yValPred, lossVal);

    freeMatrix(xBatch);
    freeMatrix(xBatchTranspose);
    freeMatrix(yBatch);
    freeMatrix(yPred);
    freeMatrix(yValPred);
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