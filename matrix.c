#include "matrix.h"

typedef struct struct_Matrix{
    double **values;
    size_t rowNumber;
    size_t columnNumber;
} Matrix;

int allocateMatrix(Matrix **mat, size_t rowNumber, size_t columnNumber){

    if(!(*mat = malloc(sizeof(Matrix)))) return 0;

    if(!rowNumber || !columnNumber){

        (*mat)->rowNumber = 0;
        (*mat)->columnNumber = 0;

        (*mat)->values = NULL;

        return 1;
        
    }

    (*mat)->rowNumber = rowNumber;
    (*mat)->columnNumber = columnNumber;

    if(!((*mat)->values = malloc(rowNumber*sizeof(double*)))){
        free(*mat);
        return 0;
    }
    for(size_t row = 0; row < rowNumber; row++){
        if(!((*mat)->values[row] = malloc(columnNumber*sizeof(double)))){

            while(row > 0){
                row--;
                free((*mat)->values[row]);
            }
            free((*mat)->values);
            free(*mat);

            return 0;

        }
    }
    
    return 1;

}

void freeMatrix(Matrix *mat){

    size_t rowNumber = mat->rowNumber;
    for(size_t row = 0; row < rowNumber; row++){
        free((mat)->values[row]);
    }

    free(mat->values);
    free(mat);

}

size_t getMatrixRowNumber(const Matrix *mat){
    return mat->rowNumber;
}

size_t getMatrixColumnNumber(const Matrix *mat){
    return mat->columnNumber;
}

int setMatrixElement(Matrix *mat, size_t row, size_t column, double element){

    if(row >= mat->rowNumber || column >= mat->columnNumber) return 0;

    mat->values[row][column] = element;

    return 1;

}

int getMatrixElement(const Matrix *mat, size_t row, size_t column, double *element){
    
    if(row >= mat->rowNumber || column >= mat->columnNumber) return 0;

    *element = mat->values[row][column];
    
    return 1;

}

int matrixTranspose(Matrix *mat, Matrix *transpose){

    if(transpose->columnNumber != mat->rowNumber) return 0;
    if(transpose->rowNumber != mat->columnNumber) return 0;

    size_t matRowNumber = mat->rowNumber;
    size_t matColumnNumber = mat->columnNumber;
    for(size_t row = 0, column; row < matRowNumber; ++row){
        for(column = 0; column < matColumnNumber; ++column){
            transpose->values[column][row] = mat->values[row][column];
        }
    }

    return 1;

}

void scalarlyMultiplyAndAssignMatrix(double scalar, Matrix *mat){
    
    size_t rowNumber = mat->rowNumber;
    size_t columnNumber = mat->columnNumber;

    for(size_t row = 0, column; row < rowNumber; ++row){
        for(column = 0; column < columnNumber; ++column){
            mat->values[row][column] *= scalar;
        }
    }

}

int subtractAndAssignMatrix(Matrix *leftMat, const Matrix *rightMat){
    
    if(leftMat->rowNumber != rightMat->rowNumber) return 0;
    if(leftMat->columnNumber != rightMat->columnNumber) return 0;

    size_t rowNumber = leftMat->rowNumber;
    size_t columnNumber = leftMat->columnNumber;

    for(size_t row = 0, column; row < rowNumber; ++row){
        for(column = 0; column < columnNumber; ++column){
            leftMat->values[row][column] -= rightMat->values[row][column];
        }
    }

    return 1;
    
}

int multiplyMatrices(const Matrix *leftMat, const Matrix *rightMat, Matrix *res){

    if(leftMat->columnNumber != rightMat->rowNumber) return 0;
    if(leftMat->rowNumber != res->rowNumber) return 0;
    if(rightMat->columnNumber != res->columnNumber) return 0;

    size_t resRowNumber = res->rowNumber;
    size_t resColumnNumber = res->columnNumber;
    size_t summandNumber = leftMat->columnNumber;
    double element, summand1, summand2;
    for(size_t row = 0, column, number; row < resRowNumber; ++row){
        for(column = 0; column < resColumnNumber; ++column){

            element = 0;
            for(number = 0; number < summandNumber; ++number){
                summand1 = leftMat->values[row][number];
                summand2 = rightMat->values[number][column];
                element += summand1 + summand2;
            }

            res->values[row][column] = element;

        }
    }

    return 1;
    
}
