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
			//double pivotCoeff = 1 / matrix[pivotRow][pivotColumn];
			//for (int j = pivotColumn; j < COLUMNS; ++j) {
			//	matrix[pivotRow][j] *= pivotCoeff;
			//}
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

void SimpleGauss::byMaxLeadColumn()
{
	std::set<int>* usedColumns = new std::set<int>();
	std::set<int>* usedRows = new std::set<int>();
	int pivotColumn = -1;
	int pivotRow = -1;
	while (usedColumns->size() + 1 != COLUMNS) {
		double maxElement = findLargestElement(pivotColumn, pivotRow, *usedColumns, *usedRows);
		if (maxElement != 0) {
			for (int i = 0; i < ROWS; ++i) {
				if (usedRows->find(i) == usedRows->end() && i != pivotRow) {
					double f = matrix[i][pivotColumn] / maxElement;
					matrix[i][pivotColumn] = 0;
					for (int j = 0; j < COLUMNS; ++j) {
						if (usedColumns->find(j) == usedColumns->end() && j != pivotColumn) {
							matrix[i][j] -= f * matrix[pivotRow][j];
						}
					}
				}
			}
		}
		std::cout << "\nPivot:" << maxElement << ", " << pivotColumn << " , " << pivotRow << "\n";

		usedRows->insert(pivotRow);
		usedColumns->insert(pivotColumn);
	}
}

double SimpleGauss::findLargestElement(int& resultColumn, int& resultRow, const std::set<int>& usedColumns, const std::set<int>& usedRows)
{
	double maxValue = 0;
	int iMax = -1;
	int jMax = -1;
	for (int i = 0; i < ROWS; ++i) {
		if (usedRows.find(i) == usedRows.end()) {
			for (int j = 0; j < COLUMNS; ++j) {
				if (usedColumns.find(j) == usedColumns.end()) {
					if (std::abs(matrix[i][j]) > std::abs(maxValue)) {
						maxValue = matrix[i][j];
						iMax = i;
						jMax = j;
					}
				}
			}
		}
	}
	resultRow = iMax;
	resultColumn = jMax;
	return maxValue;
}
