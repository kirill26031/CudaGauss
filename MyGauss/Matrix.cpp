#include "Matrix.h"

std::ostream& operator<<(std::ostream& stream, const Matrix& matrix)
{
	stream << "size: " << matrix.getHeight() << " " << matrix.getWidth();
	for (int i = 0; i < matrix.getHeight(); ++i) {
		stream << "\n";
		for (int j = 0; j < matrix.getWidth(); ++j) {
			stream << matrix[i][j];
			if (j != matrix.getWidth() - 1) {
				stream << " ";
			}
		}
	}

	return stream;
}

Matrix* readFromFile(const std::string& path)
{
	std::vector<double>* row = new std::vector<double>();
	std::vector<std::vector<double>>* matrixData = new std::vector<std::vector<double>>();
	double number;
	char delimiter;
	std::fstream in(path);
	int i = 0;
	int height;
	int width;
	std::string prefix;
	in >> prefix >> height >> width;
	while (in >> number) {
		//column delimiter
		row->push_back(number);
		if (++i == width) {
			i = 0;
			matrixData->push_back(*row);
			row = new std::vector<double>();
		}
	}
	return new Matrix(*matrixData);
}
