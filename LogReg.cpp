//
// Created by Matthew McCoy and Dmitrii Obideiko on 2/20/2023.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <utility>
#include <chrono>

using namespace std;
using namespace chrono;

void printColumnNames(const vector<string> colnames) {
    for (int i = 0; i < colnames.size(); i++) {
        cout << colnames[i] << "\t";
    }
    cout << endl;
}

double sigmoid(double z) {

    return 1.0 / (1.0 + exp(-z));

}

vector<double> logisticRegression(vector<vector<double>> data, int numIter, double learningRate) {

    int numFeatures = data[0].size() - 1;

    // initialize coefficients to 0, including bias term

    vector<double> coeffs(numFeatures + 1, 0.0);

    // start time

    auto start = high_resolution_clock::now();

    for (int iter = 0; iter < numIter; iter++) {

        double sumError = 0.0;

        for (vector<double> obs : data) {

            double y = obs[0];

            //our intercept

            double xBias = 1.0;

            // initialize with bias

            double linearPred = coeffs[0] * xBias;

            for (int j = 1; j < coeffs.size(); j++) {

                linearPred += coeffs[j] * obs[j];
            }

            double yPred = sigmoid(linearPred);
            double error = y - yPred;
            sumError += error * error;

            // update bias

            coeffs[0] += learningRate * error * yPred * (1.0 - yPred) * xBias;

            for (int j = 1; j < coeffs.size(); j++) {

                // update j-th coefficient

                coeffs[j] += learningRate * error * yPred * (1.0 - yPred) * obs[j];
            }
        }

    }

    // stop measuring training time

    auto end = high_resolution_clock::now();

    // compute duration in milliseconds

    auto trainingTime = duration_cast<milliseconds>(end - start);

    cout << "Training time: " << trainingTime.count() << " ms" << endl;

    return coeffs;
}



// Predict the class label for a single observation
double predict(vector<double> const observation, vector<double> const& coeffs) {
    double z = coeffs[0];
    for (int i = 1; i < coeffs.size(); i++) {
        z += coeffs[i] * observation[i-1];
    }
    double yHat = 1.0 / (1.0 + exp(-z));
    return (yHat >= 0.5) ? 1.0 : 0.0;
}

void evaluate(vector<vector<double>> const test, vector<double> const coeffs) {

    int tp = 0, fp = 0, tn = 0, fn = 0;

    auto start_time = steady_clock::now();

    for (auto const obs : test) {
        double y = obs[0];
        double yHat = predict(vector<double>(obs.begin() + 1, obs.end()), coeffs);

        if (y == 1.0 && yHat == 1.0) {
            tp++;
        } else if (y == 0.0 && yHat == 1.0) {
            fp++;
        } else if (y == 0.0 && yHat == 0.0) {
            tn++;
        } else if (y == 1.0 && yHat == 0.0) {
            fn++;
        }
    }
    auto end_time = steady_clock::now();
    auto diff_time = end_time - start_time;

    int numSamples = test.size();
    double accuracy = (tp + tn) / (double) numSamples;
    double sensitivity = tp / (double) (tp + fn);
    double specificity = tn / (double) (fp + tn);

    cout << "Test Metrics" << endl;
    cout << "============" << endl;
    cout << "Accuracy: " << accuracy << endl;
    cout << "Sensitivity: " << sensitivity << endl;
    cout << "Specificity: " << specificity << endl;
    cout << "Runtime: " << duration <double, milli> (diff_time).count() << " ms" << endl;
}


vector<vector<double>> stripAndConvert(const vector<vector<string>> strArray) {
        int numRows = strArray.size();
        int numCols = strArray[0].size();

        // Create the new 2d array of doubles with the correct dimensions

        vector<vector<double>> doubleArray(numRows, vector<double>(numCols - 3, 0.0));

        // Loop through each row of the original array

        for (int i = 0; i < numRows; i++) {

            // Loop through only the 3rd and 4th columns, convert to double and return it

            for (int j = 2; j < 4; j++) {
                try {
                    doubleArray[i][j - 2] = stod(strArray[i][j]);
                }
                catch (const invalid_argument e) {
                    break;
                }
                doubleArray[i][j - 2] = stod(strArray[i][j]);
            }
        }

        return doubleArray;
    }


vector<vector<double>> cleanData(const vector<vector<double>> data) {
        vector<vector<double>> cleanedData;

        for (const auto row: data) {
            bool containsNaNOrInf = false;
            for (const auto val: row) {
                if (isnan(val) || isinf(val)) {
                    containsNaNOrInf = true;
                    break;
                }
            }
            if (!containsNaNOrInf) {
                cleanedData.push_back(row);
            }
        }

        return cleanedData;
    }


pair<vector<vector<double>>, vector<vector<double>>> splitVector(const vector<vector<double>> input) {

        vector<vector<double>> first800;

        vector<vector<double>> remaining;

        if (input.size() > 800) {
            first800.assign(input.begin(), input.begin() + 800);
            remaining.assign(input.begin() + 800, input.end());
        } else {
            first800 = input;
        }

        return make_pair(first800, remaining);
    }


int main() {
        ifstream file("C:\\Users\\mattm\\GitHub\\SE-4375\\titanic_project.csv");
        //"G:\\Other computers\\My PC\\GitHub\\SE-4375\\titanic_project.csv"

        if (!file.is_open()) {
            cout << "Failed to open file." << endl;
            return 1;
        }

        vector<string> columnNames;
        vector<vector<string>> data;

        string line;

        if (getline(file, line)) {
            stringstream ss(line);
            string colName;

            while (getline(ss, colName, ',')) {
                columnNames.push_back(colName);
            }
        }

        while (getline(file, line)) {
            stringstream ss(line);
            vector<string> row;

            string colname;
            while (getline(ss, colname, ',')) {
                row.push_back(colname);
            }

            data.push_back(row);
        }

        // Print the data
        printColumnNames(columnNames);

        // Print number of rows and columns for our original data set to compare
        // This is before we clean the data to fit our needs

        cout << "Number of rows in the original data file:  " << data.size() << endl;

        cout << "Number of columns in the original data file:  " << data[0].size() << endl;

        const vector<vector<double>> cleanedData = cleanData(stripAndConvert(data));



        auto result = splitVector(cleanedData);

        vector<vector<double>> train = result.first;
        vector<vector<double>> test = result.second;

        // Print number of rows and columns for the train set after being stripped of
        //Unnecessary columns and rows and seperated into a train and a test set

        cout << "Number of rows in train:  " << train.size() << endl;

        cout << "Number of columns in train:  " << train[0].size() << endl;

        // Print number of rows and columns for the test set

        cout << "Number of rows in test: " << test.size() << endl;

        cout << "Number of columns in test: " << test[0].size() << endl;


        int numIterations = 100;
        double learningRate = 0.1;
        vector<double> coeffs = logisticRegression(train, numIterations, learningRate);
        cout << "Coefficients: ";
        for (double c: coeffs) {
            cout << c << " ";
        }
        cout << endl;

        evaluate(test, coeffs);

        return 0;


    }





