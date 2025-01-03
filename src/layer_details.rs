use crate::activation_functions::ActivationFunction;

pub struct LayerDetails {
  pub layer_size: usize,
  pub activation_function: ActivationFunction,
}

impl LayerDetails {
  pub fn new(layer_size: usize, activation_function: ActivationFunction) -> Self {
    LayerDetails {
      layer_size,
      activation_function,
    }
  }
}
