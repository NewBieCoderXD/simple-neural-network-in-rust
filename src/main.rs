use std::fmt::{Debug, Formatter};
use std::{ cell::RefCell, rc::Rc};
use activation_functions::ActivationFunction;

use crate::cost_functions::CostFunction;
use crate::neural_network_layer::NeuralNetworkLayer;
use crate::layer_details::LayerDetails;
mod neural_network_layer;
mod activation_functions;
mod layer_details;
mod cost_functions;
#[derive(Debug)]
struct NeuralNetwork<T: ActivationFunction, C: CostFunction>{
  layers: Vec<Rc<RefCell<NeuralNetworkLayer<T>>>>,
  cost_function: C
}

impl<T: ActivationFunction,C: CostFunction> NeuralNetwork<T,C>{
  fn new(layers_details: &[LayerDetails<T>], cost_function: C) -> NeuralNetwork<T,C> {
    let mut layers = vec![];
    let layers_count = layers_details.len();
    layers.reserve(layers_count);
    
    for layer_details in layers_details {
      let prev_layer = layers.last();
      let current_layer = NeuralNetworkLayer {
        bias: vec![0.0; layer_details.layer_size],
        weights: vec![Vec::new(); layer_details.layer_size],
        next_layer: None,
        prev_layer: prev_layer.cloned(),
        activation_function: layer_details.activation_function.clone()
      };

      let current_layer_rc = Rc::new(RefCell::new(current_layer));

      if let Some(layer) = prev_layer {
        layer.as_ref().borrow_mut().next_layer = Some(current_layer_rc.clone());
      };

      layers.push(current_layer_rc);
    }
    return NeuralNetwork { layers: layers, cost_function:cost_function };
  }
}

fn main() {
  let neural_network = NeuralNetwork::new(&[
    LayerDetails::new(1,activation_functions::Linear),
    LayerDetails::new(1,activation_functions::Linear)
  ], cost_functions::MSE);
  println!("{:#?}", neural_network);
}
