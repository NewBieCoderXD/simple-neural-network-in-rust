use std::fmt::Debug;

use crate::activation_functions::ActivationFunction;

#[derive(Debug)]
pub struct NeuralNetworkLayer<T: ActivationFunction> {
  pub biases: Vec<f64>,
  pub weights: Vec<Vec<f64>>,
  pub size: usize,
  // pub network: &'p NeuralNetwork<'p,T,C>,
  // pub index: usize,
  // pub next_layer: Option<Rc<RefCell<NeuralNetworkLayer<T>>>>,
  // pub prev_layer: Option<Rc<RefCell<NeuralNetworkLayer<T>>>>,
  pub activation_function: T,
  pub activations: Option<Vec<f64>>,
  pub z_values: Option<Vec<f64>>,
}

impl<T: ActivationFunction> NeuralNetworkLayer<T> {
  // pub fn forward_propagate(&mut self, network: &mut NeuralNetwork<T,C>, index: usize, prev_values: &Vec<f64>) -> Vec<f64> {
  //   // let prev_layer = &self.prev_layer.as_ref().unwrap().borrow();
  //   // let prev_values: &Vec<f64> = prev_layer.activations.as_ref().unwrap();
  //   let z = add_vector(
  //     &matrix_dot_vector(&self.weights, &prev_values),
  //     &self.biases,
  //   );
  //   let activations = T::activate(&z);
  //   self.activations = Some(activations.clone());

  //   // println!("size: {}",self.size);
  //   // println!("{:?}",prev_values);
  //   // println!("matrix dot vector: {:?}",matrix_dot_vector(&self.weights, &prev_values));
  //   // println!("z: {:?}",z);
  //   // println!("activations: {:?}", activations);

  //   if let Some(next_layer) = self.next_layer.as_mut() {
  //     return next_layer.borrow_mut().forward_propagate(&activations);
  //   }
  //   return activations;
  // }

  // fn update_weights(&mut self, error: &Vec<f64>, learning_rate: f64) {
  //   let weight_derivative = vector_dot_transposed_vector(
  //     &error,
  //     &self
  //       .prev_layer
  //       .as_ref()
  //       .unwrap()
  //       .borrow()
  //       .activations
  //       .as_ref()
  //       .unwrap(),
  //   );

  //   // println!("size: {}, weight_derivative: {:?}",self.size,weight_derivative);

  //   self.weights = add_matrices(
  //     &self.weights,
  //     &scalar_multiply_matrix(-learning_rate, &weight_derivative),
  //   );
  // }

  // fn update_biases(&mut self, error: &Vec<f64>, learning_rate: f64) {
  //   self.biases = add_vector(
  //     &self.biases,
  //     &scalar_multiply_vector(-learning_rate, &error),
  //   );
  // }

  // pub fn back_propagate<C>(&mut self, error: &Vec<f64>, learning_rate: f64)
  // where
  //   C: CostFunction,
  // {
  //   if self.prev_layer.is_none(){
  //     return;
  //   }

  //   // println!("delta {:?}",error);

  //   let next_error = matrix_dot_vector(&transpose_matrix(&self.weights), &error);

  //   Self::update_weights(self, &error, learning_rate);
  //   Self::update_biases(self, &error, learning_rate);

  //   let prev_layer = self.prev_layer.as_ref().unwrap();
  //   prev_layer.borrow_mut().back_propagate::<C>(&next_error, learning_rate);
  // }
}

// impl<T: ActivationFunction> Debug for NeuralNetworkLayer<T> {
//   fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
//     f.debug_struct("NeuralNetworkLayer")
//       .field("biases", &self.biases)
//       .field("weights", &self.weights)
//       .field("size", &self.size)
//       .field("activation_function", &self.activation_function)
//       .field("activations", &self.activations)
//       .finish()
//   }
// }
