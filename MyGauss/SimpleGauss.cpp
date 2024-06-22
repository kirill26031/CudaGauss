#include "SimpleGauss.h"

void SimpleGauss::toRowEchelonForm()
{
	while (pivotRow < ROWS && pivotColumn < COLUMNS) {
		// Find row with largest coeficient on pivot column
		int iMax = pivotRow;
		double maxValue = std::abs(matrix[iMax][pivotColumn]);
		for (int i = pivotRow + 1; i < ROWS; ++i) {
			if (std::abs(matrix[i][pivotColumn]) > maxValue) {
				maxValue = std::abs(matrix[i][pivotColumn]);
				iMax = i;
			}
		}
		
		if (maxValue == 0.0) {
			pivotColumn++;
		}
		else {
			if (pivotRow != iMax) {
				matrix.swapRows(pivotRow, iMax);
			}
			for (int i = pivotRow + 1; i < ROWS; ++i) {
				double coeff = matrix[i][pivotColumn] / matrix[pivotRow][pivotColumn];
				matrix[i][pivotColumn] = 0;
				for (int j = pivotColumn + 1; j < COLUMNS; ++j) {
					matrix[i][j] -= coeff * matrix[pivotRow][j];
				}
			}
			pivotColumn++;
			pivotRow++;
		}
	}
}
