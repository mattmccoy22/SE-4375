// Naive Bayes
// By: Dmitrii Obideiko, Matt McCoy

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>

using namespace std;

// P(A|B) = P(B|A) * P(A) / P(B)
// Posterior Probability = likelihood * class prior probability / predictor prior probability
// P(B) = probability that a person survived/died
// P(A) = probability of being in a certain class (sex, plcass, age)
// P(B|A) = prob that a person survived/died given that they were a paricular sex age/ place

class NaiveBayes {
public:
    vector<vector<string>> features;
    vector<string> target;
    vector<vector<vector<double>>> likelihoods; // probability that a person survived/not survived
    vector<vector<double>> classPriorProbs; // probabilities that a person survived based on a class (sex, age, plclass)
    vector<double> predictorPriorProb; // prob of survavial or not survival
    long trainingTime;

    NaiveBayes(vector<vector<string>> data) {
        auto dataSets = processData(data);
        this->features = dataSets.first;
        this->target = dataSets.second;
        trainModel();
    };
    
    void trainModel() {
        // Start the clock
        auto start = chrono::high_resolution_clock::now();
        
        calc_predictor_prior_prob();
        calc_class_prior_probs();
        calc_likelihoods();
        
        // End the clock
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate the duration
        trainingTime = chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    
    vector<double> calc_age_prior_prob(int age) {
        int sumAgeSurv = 0;
        int sumAgeDied = 0;
        int cntSurvived = 0;
        int cntDied = 0;
        double meanSurvived = 0;
        double varianceSurvived = 0;
        
        // count how many peopled died/survived and find the sum of all ages that correspond to those categories (died/survived)
        for (int i = 0; i < features.size(); i++) {
            if (target[i] == "1") {
                sumAgeSurv += stoi(features[i][2]);
                cntSurvived ++;
            }
            else {
                sumAgeDied += stoi(features[i][2]);
                cntDied ++;
            }
        }
        
        meanSurvived = static_cast<double>(sumAgeSurv) / cntSurvived;
        // find the sum of squares of differences from the meanSurvived
        for (int i = 0; i < features.size(); i++)
        {
            if (target[i] == "1") {
                varianceSurvived += pow(stoi(features[i][2]) - meanSurvived, 2);
            }
        }
        varianceSurvived /= (cntSurvived - 1);
        // use probability density formula
        double pSurvived =  1 / sqrt(2 * M_PI * varianceSurvived) * exp(-(pow((age - meanSurvived), 2)) / (2 * varianceSurvived));
    
        double meanDied = static_cast<double>(sumAgeDied) / cntDied;
        double varianceDied = 0;
        for (int i = 0; i < features.size(); i++)
        {
            if (target[i] == "0") {
                varianceDied += pow(stoi(features[i][2]) - meanDied, 2);
            }
        }
        varianceDied /= (cntDied - 1);
        // use probability density formula
        double pDied =  1 / sqrt(2 * M_PI * varianceDied) * exp(-(pow((age - meanDied), 2)) / (2 * varianceDied));
        
        vector<double> res = { pDied, pSurvived };
        return res;
    }
    
    void calc_predictor_prior_prob() {
        int cntSurvived = 0;
        int cntDied = 0;
        // count how many survived/died
        for (int i = 0; i < features.size(); i++) {
            if (target[i] == "1") {
                cntSurvived += 1;
            }
            else {
                cntDied ++;
            }
        }
        
        double survivedPriorProb = static_cast<double>(cntSurvived) / features.size();
        double diedPriorProb = static_cast<double>(cntDied) / features.size();
        vector<double> res = { diedPriorProb, survivedPriorProb };
        predictorPriorProb = res;
    }
    
    void calc_class_prior_probs() {
        int p1classCnt = 0;
        int p2classCnt = 0;
        int p3classCnt = 0;
        int malesCnt = 0;
        int femalesCnt = 0;
        int numOfGroups = 20;
        double ageGroups [20] = {0};
        
        // count how many males, females, people in each pclass were on the ship as well
        // as well the number of people that correspond to a particular age group
        for (int i = 0; i < features.size(); i++) {
            //plass
            if (features[i][0] == "1") {
                p1classCnt ++;
            }
            else if (features[i][0] == "2") {
                p2classCnt++;
            }
            else if (features[i][0] == "3") {
                p3classCnt++;
            }
            
            //sex
            if (features[i][1] == "0") {
                malesCnt++;
            }
            else {
                femalesCnt++;
            }
            
            int ageGroupInd = static_cast<int>(stoi(features[i][2]) / 5);
            ageGroups[ageGroupInd] += 1;
        }
        
        double p1classProb = static_cast<double>(p1classCnt) / features.size();
        double p2classProb = static_cast<double>(p2classCnt) / features.size();
        double p3classProb = static_cast<double>(p3classCnt) / features.size();
        classPriorProbs.push_back({ p1classProb, p2classProb, p3classProb });
    
        double maleProb = static_cast<double>(malesCnt) / features.size();
        double femaleProb = static_cast<double>(femalesCnt) / features.size();
        classPriorProbs.push_back({ maleProb, femaleProb });
        
        vector<double> ageGroupProbs;
        for (int i = 0; i < numOfGroups; i++) {
            ageGroupProbs.push_back(static_cast<double>(ageGroups[i]) / features.size());
        }
        classPriorProbs.push_back(ageGroupProbs);
    }
    
    void calc_likelihoods() {
        // Class
        int c1CntSurvived  = 0;
        int c1CntTotal = 0;
        int c2CntSurvived  = 0;
        int c2CntTotal = 0;
        int c3CntSurvived  = 0;
        int c3CntTotal = 0;
        int c1CntDied = 0;
        int c2CntDied = 0;
        int c3CntDied = 0;
        
        // count the numer number for each class and how many people died in each class
        for (int i = 0; i < features.size(); i++) {
            if (features[i][0] == "1") {
                if (target[i] == "1") {
                    c1CntSurvived ++;
                }
                else {
                    c1CntDied++;
                }
                c1CntTotal += 1;
            }
            else if (features[i][0] == "2") {
                if (target[i] == "1") {
                    c2CntSurvived ++;
                }
                else {
                    c2CntDied++;
                }
                c2CntTotal += 1;
            }
            else {
                if (target[i] == "1") {
                    c3CntSurvived ++;
                }
                else {
                    c3CntDied++;
                }
                c3CntTotal += 1;
            }
        }
        
        // find probability of the person surviving based on their class
        double c1ProbSurv = static_cast<double>(c1CntSurvived) / c1CntTotal;
        double c2ProbSurv = static_cast<double>(c2CntSurvived) / c2CntTotal;
        double c3ProbSurv = static_cast<double>(c3CntSurvived) / c3CntTotal;
    
        // find probability of the person dying based on their class
        double c1ProbDied = static_cast<double>(c1CntDied) / c1CntTotal;
        double c2ProbDied = static_cast<double>(c2CntDied) / c2CntTotal;
        double c3ProbDied = static_cast<double>(c3CntDied) / c3CntTotal;
        vector<double> pclassProbsSurv = { c1ProbSurv, c2ProbSurv, c3ProbSurv };
        vector<double> pclassProbsDied = { c1ProbDied, c2ProbDied, c3ProbDied };
        
        vector<vector<double>> pclassProbs = { pclassProbsDied, pclassProbsSurv };
        likelihoods.push_back(pclassProbs);
        
        // Sex
        int femaleCntSurvived  = 0;
        int femaleCntTotal = 0;
        int maleCntSurvived    = 0;
        int maleCntTotal   = 0;
        int femaleCntDied = 0;
        int maleCntDied = 0;
        for (int i = 0; i < features.size(); i++) {
            if (features[i][1] == "1") {
                if (target[i] == "1") {
                    maleCntSurvived ++;
                }
                else {
                    maleCntDied++;
                }
                maleCntTotal ++;
            }
            else {
                if (target[i] == "1") {
                    femaleCntSurvived ++;
                }
                else {
                    femaleCntDied ++;
                }
                femaleCntTotal ++;
            }
        }
        
        double maleProbSurv = static_cast<double>(maleCntSurvived) / maleCntTotal;
        double femaleProbSurv = static_cast<double>(femaleCntSurvived) / femaleCntTotal;
        double maleProbDied = static_cast<double>(maleCntDied) / maleCntTotal;
        double femaleProbDied = static_cast<double>(femaleCntDied) / femaleCntTotal;
        vector<double> sexProbsSurv = { femaleProbSurv, maleProbSurv };
        vector<double> sexProbsDied = { femaleProbDied, maleProbDied };
        
        vector<vector<double>> sexProbs = { sexProbsDied, sexProbsSurv};
        likelihoods.push_back(sexProbs);
    }
            
    string predict(string pclass, string sex, string age) {
        //P(survived | age, sex, pclass) = P(age | survived) * P(sex | survived) * P(pclass | survived) * P(survived) / (P(age) * P(sex) * P(pclass))
        
        // there's 20 groups in total, so we divide the age by 5 (assuming that max age to live is 100)
        int ageGroupInd = static_cast<int>(stoi(age) / 5);
        
        double pDied = (
                    likelihoods[0][0][stoi(pclass) - 1] *
                    likelihoods[1][0][stoi(sex)] *
                    calc_age_prior_prob(stoi(age))[0] *
                    predictorPriorProb[0] /
                    classPriorProbs[0][stoi(pclass) - 1] /
                    classPriorProbs[1][stoi(sex)] /
                    classPriorProbs[2][ageGroupInd]
                    );
        
        double pSurvived = (
                    likelihoods[0][1][stoi(pclass) - 1] *
                    likelihoods[1][1][stoi(sex)] *
                    calc_age_prior_prob(stoi(age))[1] *
                    predictorPriorProb[1] /
                    classPriorProbs[0][stoi(pclass) - 1] /
                    classPriorProbs[1][stoi(sex)] /
                    classPriorProbs[2][ageGroupInd]
                    );
        
        // return 1 is the person is more likely to survive; 0 otherwise
        if (pSurvived > pDied) {
            return "1";
        }
        else {
            return "0";
        }
    }
    
    // clean up data and break it into features and target sets
    pair<vector<vector<string>>, vector<string> >processData(vector<vector<string>> data) {
        vector<vector<string>> features = data;
        for (int i = 0; i < features.size(); i++) {
            // remove third column - the "survived" column
            features[i].erase(features[i].begin() + 2);
            // remove first column the "" column (the column doesn't have a name)
            features[i].erase(features[i].begin()); // remove first element
            // clean up - remove last 2 characters
            features[i][2] = features[i][2].substr(0, features[i][2].size() - 1);
        }
        
        // add values fromo the "survived" column to the target
        vector<string> target;
        for (int i = 0; i < features.size(); i++) {
            target.push_back(data[i][2]);
        }
        
        return make_pair(features, target);
    }
    
    // evaluat the performance of the algorithm
    void evaluate(vector<vector<string>> testData) {
        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;
        auto dataSets = processData(testData);
        vector<vector<string>> testFeatures = dataSets.first;
        vector<string> testTarget = dataSets.second;
        
        for (int i = 0; i < testFeatures.size(); i++) {
            string prediction = predict(testFeatures[i][0], testFeatures[i][1], testFeatures[i][2]);
            if (prediction == "1") {
                if (prediction == testTarget[i]) {
                    tp ++;
                }
                else {
                    fp ++;
                }
            }
            else {
                if (prediction == testTarget[i]) {
                    tn ++;
                }
                else {
                    fn ++;
                }
            }
        }
        
        // how well the model can detedct the positive cases (the person survived)
        double sensitivity = static_cast<double>(tp) / (tp + fn);
        // how well the model can detect negative cases (the person died)
        double specificity = static_cast<double>(tn) / (tn + fp);
        // measures how correctly teh model can predict the class of the label
        double accuracy = static_cast<double>(tp + tn) / (tp + tn + fp + fn);
        
        cout<< "__NaiveBayes Performance__" << endl;
        cout << "Training Time: " << trainingTime << " microseconds" << endl;
        cout << "Sensitivity: " << round(sensitivity * 100) / 100 << endl;
        cout << "Specificity: " << round(specificity * 100) / 100 << endl;
        cout << "Accuracy: " << round(accuracy * 100) / 100 << endl;
    }
};

int main(int argc, const char * argv[]) {
    // check if such file exists
    ifstream file("titanic_project.csv");

        if (!file.is_open()) {
        cout << "Failed to open file." << endl;
        return 1;
    }
    
    // Read file and store the data
    vector<string> colNames;
    vector<vector<string>> data;

    string line;
    if (getline(file, line)) {
        // break the line into words
        stringstream ss(line);
        string colName;
        
        // add all the column names to the list
        while (getline(ss, colName, ',')) {
            colNames.push_back(colName);
        }
    }
    
    while (getline(file, line)) {
        // break the line into words
        stringstream ss(line);
        vector<string> row;

        string col;
        // push all values into a row
        while (getline(ss, col, ',')) {
            row.push_back(col);
        }

        data.push_back(row);
    }
    
    // split data into training and testing
    vector<vector<string>> trainingData(data.begin(), data.begin() + 800);
    vector<vector<string>> testingData(data.begin() + 800, data.end());
    
    // run the naive bayes algorithm and evaluate its performance
    NaiveBayes* obj = new NaiveBayes(trainingData);
    obj->evaluate(testingData);
    
    return 0;
}
