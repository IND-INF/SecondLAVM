#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdlib.h>

typedef struct struct_Matrix Matrix;

int allocateMatrix(Matrix **mat, size_t rowNumber, size_t columnNumber);
void freeMatrix(Matrix *mat);

size_t getMatrixRowNumber(const Matrix *mat);
size_t getMatrixColumnNumber(const Matrix *mat);

int setMatrixElement(Matrix *mat, size_t row, size_t column, double element);
int getMatrixElement(const Matrix *mat, size_t row, size_t column, double *element);

int matrixTranspose(Matrix *mat, Matrix *transpose);

void scalarlyMultiplyAndAssignMatrix(double scalar, Matrix *mat);
int subtractAndAssignMatrix(Matrix *leftMat, const Matrix *rightMat);

int multiplyMatrices(const Matrix *leftMat, const Matrix *rightMat, Matrix *res);

#endif //_MATRIX_H_