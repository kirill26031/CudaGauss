#pragma once
#include <ostream>
#include <fstream>
#include <vector>
#include <iostream>
class Matrix
{
private: 
	int width;
	int height;
	double** data;

public:
	Matrix(int height, int width, double** data = nullptr) : width(width), height(height) {
		this->data = new double* [height];
		for (int i = 0; i < height; ++i) {
			this->data[i] = new double[width];
			for (int j = 0; j < width; ++j) {
				if (data == nullptr) {
					this->data[i][j] = 0;
				}
				else {
					this->data[i][j] = data[i][j];
				}
			}
		}
	}

	Matrix(const Matrix& matrix) : height(matrix.getHeight()), width(matrix.getWidth()), data(matrix.data) {}

	Matrix(const std::vector<std::vector<double>>& vectorData) : height(vectorData.size()), width(vectorData.size() > 0 ? vectorData[0].size() : 0), data(new double* [height]) {
		for (int i = 0; i < height; ++i) {
			data[i] = new double[width];
			for (int j = 0; j < width; ++j) {
				data[i][j] = vectorData[i][j];
			}
		}
	}

	~Matrix() {
		delete[] this->data;
	}

	// Returns row
	double * operator[](int i) const {
		return this->data[i];
	}

	int const& getHeight() const {
		return this->height;
	}

	int const& getWidth() const {
		return this->width;
	}

	void swapRows(int row_a, int row_b) {
		if (row_a >= 0 && row_b >= 0 && row_a < height && row_b < height && row_a != row_b) {
			double* temp = data[row_a];
			data[row_a] = data[row_b];
			data[row_b] = temp;
		}
		else {
			std::cerr << "swapRows " << row_a << ", " << row_b << std::endl;
		}
	}
};

std::ostream& operator<< (std::ostream& stream, const Matrix& matrix);

Matrix* readFromFile(const std::string& path);

