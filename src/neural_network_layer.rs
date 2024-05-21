use std::{cell::RefCell, rc::Rc};
use std::fmt::{Debug, Formatter};

use crate::activation_functions::ActivationFunction;

// #[derive(Debug)]
pub struct NeuralNetworkLayer<T:ActivationFunction>{
  pub bias: Vec<f64>,
  pub weights: Vec<Vec<f64>>,
  pub next_layer: Option<Rc<RefCell<NeuralNetworkLayer<T>>>>,
  pub prev_layer: Option<Rc<RefCell<NeuralNetworkLayer<T>>>>,
  pub activation_function: T
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
      .field("bias", &self.bias)
      .field("weights", &self.weights)
      .field("next_layer", &next_layer_ptr)
      .field("prev_layer", &prev_layer_ptr)
      .finish()
  }
}