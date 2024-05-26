use std::{cell::{Ref, RefCell}, rc::Rc};
use rand::Rng;
use crate::{
  activation_functions::ActivationFunction, cost_functions::CostFunction, hadamard_product, layer_details::LayerDetails, neural_network_layer::NeuralNetworkLayer
};
use std::fmt::Debug;

pub struct NeuralNetwork<T: ActivationFunction, C: CostFunction> {
  layers: Vec<Rc<RefCell<NeuralNetworkLayer<T>>>>,
  cost_function: C,
}

impl<T: ActivationFunction, C: CostFunction> Debug for NeuralNetwork<T, C> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f
      .debug_struct("NeuralNetwork")
      .field(
        "layers",
        &self.layers.iter().map(|layer| {
          layer.as_ref().borrow()
        }).collect::<Vec<Ref<NeuralNetworkLayer<T>>>>(),
      )
      .field("cost_function", &self.cost_function)
      .finish()
  }
}

impl<T: ActivationFunction, C: CostFunction> NeuralNetwork<T, C> {
  fn random_vector(size: usize) -> Vec<f64>{
    (0..size).map(|_index|{
      rand::thread_rng().gen_range(0.01..0.99)
    }).collect()
  }
  pub fn new(layers_details: &[LayerDetails<T>], cost_function: C) -> NeuralNetwork<T, C> {
    assert!(
      layers_details.len() >= 2,
      "There are not enough layers to function({} layers), please make at least 2",
      layers_details.len()
    );
    let mut layers: Vec<Rc<RefCell<NeuralNetworkLayer<T>>>> = vec![];
    let layers_count = layers_details.len();
    layers.reserve(layers_count);

    for layer_details in layers_details {
      let prev_layer = layers.last();
      let weights;
      let biases;
      if let Some(prev_layer) = prev_layer {
        weights = (0..layer_details.layer_size).map(|_|Self::random_vector(prev_layer.borrow().size)).collect();
        biases = vec![0.0; layer_details.layer_size];
      } else {
        weights = vec![];
        biases = vec![];
      }
      let current_layer = NeuralNetworkLayer {
        biases,
        weights,
        size: layer_details.layer_size,
        next_layer: None,
        prev_layer: prev_layer.cloned(),
        activation_function: layer_details.activation_function.clone(),
        activations: None,
      };

      let current_layer_rc = Rc::new(RefCell::new(current_layer));

      if let Some(layer) = prev_layer {
        layer.as_ref().borrow_mut().next_layer = Some(current_layer_rc.clone());
      };

      layers.push(current_layer_rc);
    }
    return NeuralNetwork {
      layers: layers,
      cost_function: cost_function,
    };
  }

  pub fn forward_propagate(&self, input: &Vec<f64>) -> Vec<f64> {
    assert_eq!(input.len(), self.layers[0].borrow().size);
    self.layers[0].borrow_mut().activations=Some(input.to_vec());
    return self.layers[1].borrow_mut().forward_propagate(&input);
  }

  pub fn back_propagate(&self, actual: &Vec<f64>, learning_rate: f64) {
    let mut last_layer = self.layers.last().unwrap().borrow_mut();
    assert_eq!(actual.len(), last_layer.size);
    
    let activations = last_layer.activations.as_ref().unwrap();
    let cost_derivative = C::derivative(&activations, actual);
    let activation_derivative = T::derivative(&activations);
    let error = hadamard_product!(cost_derivative, activation_derivative);

    let cost = C::cost(actual, activations.as_ref());
    println!("cost: {}", cost);
    // println!("cost derivation: {:?}", cost_derivative);

    last_layer
      .back_propagate::<C>(&error, learning_rate);
  }
}
