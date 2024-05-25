use std::fmt::{Debug, Formatter};
use std::{cell::RefCell, rc::Rc};

use crate::activation_functions::ActivationFunction;
use crate::cost_functions::CostFunction;
use crate::linear_algebra::{
  add_matrices, add_vector, hadamard_product, matrix_dot_vector, scalar_multiply_matrix, scalar_multiply_vector, vector_dot_transposed_vector
};

use crate::hadamard_product;
// #[derive(Debug)]
pub struct NeuralNetworkLayer<T: ActivationFunction> {
  pub biases: Vec<f64>,
  pub weights: Vec<Vec<f64>>,
  pub size: usize,
  pub next_layer: Option<Rc<RefCell<NeuralNetworkLayer<T>>>>,
  pub prev_layer: Option<Rc<RefCell<NeuralNetworkLayer<T>>>>,
  pub activation_function: T,
  pub activations: Option<Vec<f64>>,
}

impl<T: ActivationFunction> NeuralNetworkLayer<T> {
  pub fn forward_propagate(&mut self, prev_values: &Vec<f64>) -> Vec<f64> {
    // let prev_layer = &self.prev_layer.as_ref().unwrap().borrow();
    // let prev_values: &Vec<f64> = prev_layer.activations.as_ref().unwrap();
    let z = add_vector(
      &matrix_dot_vector(&self.weights, &prev_values),
      &self.biases,
    );
    let activations = T::activate(&z);
    self.activations = Some(activations.clone());

    // println!("size: {}",self.size);
    // println!("{:?}",prev_values);
    // println!("matrix dot vector: {:?}",matrix_dot_vector(&self.weights, &prev_values));
    // println!("z: {:?}",z);
    println!("activations: {:?}",activations);

    if let Some(next_layer) = self.next_layer.as_mut(){
      return next_layer.borrow_mut().forward_propagate(&activations);
    }
    return activations;
  }

  fn update_weights(&mut self, error: &Vec<f64>, learning_rate: f64) {
    let weight_derivative = vector_dot_transposed_vector(
      &error,
      &self
        .prev_layer
        .as_ref()
        .unwrap()
        .borrow()
        .activations
        .as_ref()
        .unwrap(),
    );
    self.weights = add_matrices(
      &self.weights,
      &scalar_multiply_matrix(-learning_rate, &weight_derivative),
    );
  }

  fn update_biases(&mut self, error: &Vec<f64>, learning_rate: f64) {
    self.biases = add_vector(
      &self.biases,
      &scalar_multiply_vector(-learning_rate, &error),
    );
  }

  pub fn back_propagate<C>(&mut self, actual: &Vec<f64>, learning_rate: f64)
  where
    C: CostFunction,
  {
    assert!(self.activations.is_some());
    let activations = self.activations.as_mut().unwrap();
    let cost_derivative = C::derivative(&activations, actual);
    let activation_derivative = T::derivative(&activations);
    let cost = C::cost(actual, activations.as_ref());
    println!("cost: {}", cost);
    // let error = hadamard_product(&[cost_derivative,activation_derivative]);
    let error = hadamard_product!(cost_derivative, activation_derivative);
    println!("{:?}", activation_derivative);
    Self::update_weights(self, &error, learning_rate);
    Self::update_biases(self, &error, learning_rate);
  }
}

impl<T: ActivationFunction> Debug for NeuralNetworkLayer<T> {
  fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
    // let gg =self.next_layer.as_ref().unwrap().borrow().;
    let next_layer_ptr;
    if let Some(next_layer) = self.next_layer.as_ref() {
      next_layer_ptr = format!("{:p}", &*next_layer.borrow());
    } else {
      next_layer_ptr = "0".to_string();
    }
    let prev_layer_ptr;
    if let Some(prev_layer) = self.prev_layer.as_ref() {
      prev_layer_ptr = format!("{:p}", &*prev_layer.borrow());
    } else {
      prev_layer_ptr = "0".to_string();
    }
    f.debug_struct("NeuralNetworkLayer")
      .field("address", &format!("{:p}", self))
      .field("biases", &self.biases)
      .field("weights", &self.weights)
      .field("size", &self.size)
      .field("activation_function", &self.activation_function)
      .field("next_layer", &next_layer_ptr)
      .field("prev_layer", &prev_layer_ptr)
      .finish()
  }
}
