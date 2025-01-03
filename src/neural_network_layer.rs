use std::fmt::Debug;

use crate::activation_functions::ActivationFunction;

#[derive(Debug)]
pub struct NeuralNetworkLayer {
  pub biases: Vec<f64>,
  pub weights: Vec<Vec<f64>>,
  pub size: usize,
  pub activation_function: ActivationFunction,
  pub activations: Option<Vec<f64>>,
  pub z_values: Option<Vec<f64>>,
}
