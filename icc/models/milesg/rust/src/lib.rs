#![feature(proc_macro, specialization)]

extern crate rusty_machine as rm;
extern crate pyo3;

use pyo3::prelude::*;

use rm::learning::logistic_reg::LogisticRegressor;
use rm::learning::SupModel;
use rm::linalg::{Matrix, Vector};
use std::vec::Vec;
use std::option::Option;



#[py::modinit(models)]
fn init_mod(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "logistic_regression")]
    fn logistic_regression_py(mut xTest: Vec<Vec<f64>>,
                              mut xTrain: Vec<Vec<f64>>,
                              mut yTrain: Vec<f64>) -> PyResult<Option<Vec<f64>>> {
        /*
            Python interface to the logistic regression model
        */
        let result = train_n_predict_log(xTest, xTrain, yTrain);
        Ok(result)
    }

    Ok(())
}


fn train_n_predict_log(mut xTest: Vec<Vec<f64>>,
                       mut xTrain: Vec<Vec<f64>>,
                       mut yTrain: Vec<f64>) -> Option<Vec<f64>> {
    /*
        Implement training and predicting of a vanilla Logistic Regression model
    */
    let xTest: Matrix<f64> = convert_x_to_matrix(xTest);
    let xTrain: Matrix<f64> = convert_x_to_matrix(xTrain);
    let yTrain: Vector<f64> = Vector::new(yTrain);

    let mut model = LogisticRegressor::default();

    // Train and predict if training was successful
    if let Err(err) = model.train(&xTrain, &yTrain) {
        println!("Error training model: {}", err);
        None

    } else {

        // Training was successful, now predict on test
        if let Ok(out) = model.predict(&xTest) {

            // Return rm::linalg::Vector as std::vec::Vector
            Some(out.iter().map(|x: &f64| *x).collect::<Vec<f64>>())

        // Prediction failed...
        } else {
            None
        }
    }
}


fn convert_x_to_matrix(x: Vec<Vec<f64>>) -> Matrix<f64> {
    /*
        Convert the passed nested vectors to the rusty_machine Matrix type
    */
    let (n_rows, n_cols) = (x.len(), x[0].len());
    let mut inputs = Vec::with_capacity(n_rows * n_cols);
    for row in x {
        inputs.extend(row);
    }
    Matrix::new(n_rows, n_cols, inputs)
}

