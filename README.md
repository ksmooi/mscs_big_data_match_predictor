
# Match Predictor

The Match Predictor is a soccer match prediction app that uses various machine learning models to predict match outcomes. The system consists of a backend written in Python using Flask and a frontend written in TypeScript and React.

## Key Components

### Frontend
- Allows users to interact with the app by selecting teams, leagues, and prediction models.
- Displays predicted match outcomes and confidence levels.

### Backend
- Handles the logic for predicting match outcomes.
- Consists of models and predictors conforming to a common interface with a `predict` method.

## Setting Up the App

1. Use terminal commands to run both the frontend and backend.
2. Interact with the app through the frontend, choosing leagues, teams, and models to see predictions.

## Predictors

### Simple Models
- **Home Predictor**: Always predicts the home team to win.
- **Alphabet Predictor**: Uses alphabetical order of team names to predict outcomes.

### Advanced Models
- **Form-Based Predictor**: Uses recent match outcomes to predict the result.
- **Past Results Predictor**: Uses historical match data to predict outcomes based on past performance.
- **Linear Regression Predictor**: Uses scikit-learn for linear regression-based predictions.
- **Enhanced Logistic Regression Predictor**: Uses logistic regression with additional features.
- **Enhanced Gradient Boosting Predictor**: Uses gradient boosting with additional features.
- **Simulation Predictors**: Simulate matches based on team scoring probabilities.

## Data Handling

- The app uses historical data from various soccer leagues to train the models.
- The data includes match fixtures, outcomes, and goals.

## Testing and Improving Models

- **Model Tests**: Ensure each model meets a minimum accuracy threshold.
- **Predictor Tests**: Validate the accuracy of the predictors using historical data.
- **Measure Task**: Runs detailed accuracy tests for models that require training.
- **Report Task**: Evaluates model performance across different leagues and seasons.

## Goals for Users

- Understand the existing models and predictors.
- Create and improve their own models to achieve higher prediction accuracy.
- Use testing and reporting tools to iteratively refine their models.

## Final Notes

- The course encourages students to explore and enhance the match predictor models.
- Users are invited to share their high-performing models for feedback and potential inclusion in the course materials.

## Model Performance Results

### Barclays Premier League 2021

| Predictor                    | Accuracy | Elapsed     |
|------------------------------|----------|-------------|
| Home                         | 0.428947 | 0.000260s   |
| Points                       | 0.539474 | 0.000421s   |
| Offense simulator (fast)     | 0.500000 | 2.499645s   |
| Offense simulator            | 0.507895 | 24.994169s  |
| Full simulator (fast)        | 0.542105 | 3.304595s   |
| Full simulator               | 0.536842 | 32.963316s  |
| **Enhanced Logistic Regression** | 0.526316 | 0.175424s   |
| **Enhanced Gradient Boosting**   | 0.507895 | 0.296986s   |
| Form-Based                   | 0.471053 | 0.001195s   |
| Alphabet                     | 0.400000 | 0.000268s   |

### English League Championship 2021

| Predictor                    | Accuracy | Elapsed     |
|------------------------------|----------|-------------|
| Home                         | 0.443447 | 0.000340s   |
| Points                       | 0.368043 | 0.000475s   |
| Offense simulator (fast)     | 0.394973 | 3.650437s   |
| Offense simulator            | 0.411131 | 36.564387s  |
| Full simulator (fast)        | 0.416517 | 4.811156s   |
| Full simulator               | 0.414722 | 48.000090s  |
| **Enhanced Logistic Regression** | 0.402154 | 0.248842s   |
| **Enhanced Gradient Boosting**   | 0.423698 | 0.531531s   |
| Form-Based                   | 0.414722 | 0.001605s   |
| Alphabet                     | 0.378815 | 0.000421s   |

### Italy Serie A 2021

| Predictor                    | Accuracy | Elapsed     |
|------------------------------|----------|-------------|
| Home                         | 0.389474 | 0.000254s   |
| Points                       | 0.507895 | 0.000366s   |
| Offense simulator (fast)     | 0.494737 | 2.496860s   |
| Offense simulator            | 0.500000 | 24.937453s  |
| Full simulator (fast)        | 0.523684 | 3.262647s   |
| Full simulator               | 0.523684 | 32.446993s  |
| **Enhanced Logistic Regression** | 0.518421 | 0.160753s   |
| **Enhanced Gradient Boosting**   | 0.502632 | 0.269050s   |
| Form-Based                   | 0.468421 | 0.001041s   |
| Alphabet                     | 0.400000 | 0.000294s   |


## How to Run

### Prerequisites

- Ensure you have Python and Node.js installed on your machine.
- Install the required Python packages:
  ```bash
  pip install -r requirements.txt
  ```
- Install the required Node.js packages:
  ```bash
  npm install
  ```

### Running the App

1. Start the backend server:
   ```bash
   make run-backend
   ```
2. Start the frontend server:
   ```bash
   make run-frontend
   ```
3. Open your browser and go to `http://localhost:3000` to interact with the app.

### Running Tests

- To run unit tests:
  ```bash
  make backend/test
  ```
- To run detailed accuracy tests:
  ```bash
  make measure
  ```
- To generate a performance report:
  ```bash
  make backend/report
  ```

## Directory Structure

```
backend/
├── matchpredictor/
│   ├── predictors/
│   │   ├── alphabet_predictor.py
│   │   ├── enhanced_gradient_boosting_predictor.py
│   │   ├── enhanced_logistic_regression_predictor.py
│   │   ├── form_based_predictor.py
│   │   └── ...
│   ├── ...
│   ├── app.py
│   └── ...
└── ...
├── test/
│   ├── predictors/
│   │   ├── measure_alphabet_predictor.py
│   │   ├── measure_enhanced_gradient_boosting_predictor.py
│   │   ├── measure_enhanced_logistic_regression_predictor.py
│   │   ├── measure_form_based_predictor.py
│   │   └── ...
│   └── ...
frontend/
├── src/
│   ├── components/
│   ├── App.tsx
│   ├── index.tsx
│   └── ...
└── ...

```


## Installation

Follow the instructions below to get the app up and running on your machine.

1.  Install Python 3.10 and a recent version of NPM.
2.  Install dependencies and run tests.
    ```shell
    make install test
    ```
3.  View the list of available tasks
    ```shell
    make
    ```

### Backend

Here are a few tasks that are useful when running the backend app.
Make sure they all run on your machine.

1.  Run tests
    ```shell
    make backend/test

1.  Run model measurement tests
    ```shell
    make backend/measure
    ```

2.  Run server
    ```shell
    make backend/run
    ```

3.  Run an accuracy report
    ```shell
    make backend/report
    ```

### Frontend

Here are a few tasks that are useful when running the frontend app.
Make sure they all run on your machine.

1.  Run tests
    ```shell
    make frontend/test
    ```

2.  Run server
    ```shell
    make frontend/run
    ```

## Integration tests

If it's helpful, you may want to run integration tests during development.
Do so with the tasks below.

1.  Run tests
    ```shell
    make integration/test
    ```

2.  Interactive mode
    ```shell
    make integration/run
    ```
