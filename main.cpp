#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;



double sum(vector<double> vector1);

double mean(vector<double> vector1);

double median(vector<double> vector1);

double range(vector<double> vector1);

double cov(vector<double> vector1, vector<double> vector2);

double covsum(vector<double> vector1, vector<double> vector2);

double corr(vector<double> vector1, vector<double> vector2);

double corrVar(vector<double> vector1);

void printStats(vector<double> vector1);

ifstream inDoc;
string line;
string rm_in, medv_in;
const int MAX_LEN = 1000;
vector<double> rm(MAX_LEN);
vector<double> medv(MAX_LEN);
int numObservations = 0;

int main(int argc, char** argv) {


    cout << "Hello, World!" << endl;

    //open the file
    inDoc.open("Boston.csv");

    //make sure it opened
    if(!inDoc.is_open()) {
        cout << "not open" << endl;
    }

    //get first line with "rm" and "medv"
    getline(inDoc, line);



    //fill up the vectors
    while (inDoc.good()) {
        getline(inDoc, rm_in, ',');

        getline(inDoc, medv_in, '\n');


        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);

        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);



    cout << "Number of Records: " << numObservations << endl;

    cout << "Stats for rm: " << endl;
    printStats(rm);

    cout << "Stats for medv: " << endl;
    printStats(medv);

    cout << "Covariance of rm and medv: "<< cov(rm, medv) << endl;

    cout << "Correlation of rm and medv: "<< corr(rm, medv) << endl;


    return 0;



}


double range(vector<double> vector1) {
    double min = 10, max, rng;

    //go through each vector updating when you find a larger or smaller element
    for(int i = 0; i < numObservations; i++){
        if (vector1[i] < min)
            min = vector1[i];
        if (vector1[i] > max)
            max = vector1[i];

    }


    rng = max - min;
    return rng;
}

double median(vector<double> vector1) {
    double med;

    med = vector1[vector1.size()/2 - 1];

    return med;
}

double mean(vector<double> vector1) {
    double mn;

    mn = sum(vector1)/vector1.size();

    return mn;
}

double sum(vector<double> vector1) {
    double total = 0;

    for(int i = 0; i < numObservations; i++){
        total = vector1[i] + total;
    }

    return total;

}

//Covariance
double cov(vector<double> vector1, vector<double> vector2){

    double cov = (covsum(vector1, vector2)/(numObservations -1));

    return cov;
}


//helper function to find the sum
double covsum(vector<double> vector1, vector<double> vector2){
    double total = 0;

    for(int i = 0; i < vector1.size() && i < vector2.size(); i++){


        double x = vector1[i] - mean(vector1);
        double y = vector2[i] - mean(vector2);
        double z = x * y;

        total = total + z;

    }

    return total;
}

//Correlation
double corr(vector<double> vector1, vector<double> vector2){

    double denominator =sqrt((corrVar(vector1))*(corrVar(vector2)));

    double corr = cov(vector1, vector2)/denominator;

    return corr;
}

//helper function to find variance
double corrVar(vector<double> vector1){

    double total = 0;

    for(int i = 0; i < vector1.size(); i++) {

        double x = vector1[i] - mean(vector1);

        x = x*x;

        total = total + x;
    }

    total = total/(vector1.size() - 1);

    return total;

}

//function to print all 4 basic stats
void printStats(vector<double> vector1) {

    cout << "Sum: " << sum(vector1) << endl;


    cout << "Mean: " << mean(vector1) << endl;


    cout << "Median: " << median(vector1) << endl;


    cout << "Range: " << range(vector1) << endl;


}


