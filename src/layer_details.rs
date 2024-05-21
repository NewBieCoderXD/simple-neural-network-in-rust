use crate::activation_functions::ActivationFunction;

pub struct LayerDetails<F>
where F: ActivationFunction
{
  pub layer_size: usize,
  pub activation_function: F
}

impl<F> LayerDetails<F>
where F: ActivationFunction
{
  pub fn new(layer_size: usize, activation_function: F) -> Self{
    LayerDetails{
      layer_size,
      activation_function
    }
  }
}