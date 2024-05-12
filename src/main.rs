use std::fmt::{Debug, Formatter};
use std::{borrow::BorrowMut, cell::RefCell, rc::Rc};

// #[derive(Debug)]
struct NeuralNetworkLayer {
  bias: Vec<f64>,
  weights: Vec<Vec<f64>>,
  next_layer: Option<Rc<RefCell<NeuralNetworkLayer>>>,
  prev_layer: Option<Rc<RefCell<NeuralNetworkLayer>>>,
}

impl Debug for NeuralNetworkLayer {
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

#[derive(Debug)]
struct NeuralNetwork {
  layers: Vec<Rc<RefCell<NeuralNetworkLayer>>>,
}

impl NeuralNetwork {
  fn new(layer_sizes: &[usize]) -> NeuralNetwork {
    let mut layers = vec![];
    let layers_count = layer_sizes.len();
    layers.reserve(layers_count);

    for &layer_size in layer_sizes {
      let prev_layer = layers.last();
      let current_layer = NeuralNetworkLayer {
        bias: vec![0.0; layer_size],
        weights: vec![Vec::new(); layer_size],
        next_layer: None,
        prev_layer: prev_layer.cloned(),
      };

      let current_layer_rc = Rc::new(RefCell::new(current_layer));

      if let Some(layer) = prev_layer {
        layer.as_ref().borrow_mut().next_layer = Some(current_layer_rc.clone());
      };

      layers.push(current_layer_rc);
    }
    return NeuralNetwork { layers: layers };
  }
}

fn main() {
  let neural_network = NeuralNetwork::new(&[1, 1]);
  println!("{:#?}", neural_network);
}
