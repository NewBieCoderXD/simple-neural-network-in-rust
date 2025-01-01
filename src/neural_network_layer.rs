use std::fmt::Debug;

use crate::activation_functions::ActivationFunction;

#[derive(Debug)]
pub struct NeuralNetworkLayer {
  pub biases: Vec<f64>,
  pub weights: Vec<Vec<f64>>,
  pub size: usize,
  // pub network: &'p NeuralNetwork<'p,T,C>,
  // pub index: usize,
  // pub next_layer: Option<Rc<RefCell<NeuralNetworkLayer<T>>>>,
  // pub prev_layer: Option<Rc<RefCell<NeuralNetworkLayer<T>>>>,
  pub activation_function: ActivationFunction,
  pub activations: Option<Vec<f64>>,
  pub z_values: Option<Vec<f64>>,
}