#pragma once
#ifndef __MATRIX
#define __MATRIX
#include "Matrix.h"
#endif

#include <cmath>

class SimpleGauss
{
public:
	int ROWS;
	int COLUMNS;
	int pivotRow;
	int pivotColumn;
	Matrix& matrix;

	SimpleGauss(Matrix& matrix) :
		ROWS(matrix.getHeight()), COLUMNS(matrix.getWidth()), pivotRow(0), pivotColumn(0), matrix(matrix) {}

	SimpleGauss(const SimpleGauss& copy) : pivotColumn(copy.pivotColumn), pivotRow(copy.pivotRow),
		ROWS(copy.ROWS), COLUMNS(copy.COLUMNS), matrix(copy.matrix) {}

	void toRowEchelonForm();
};

