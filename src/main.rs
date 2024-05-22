use neural_network::NeuralNetwork;
use crate::{layer_details::LayerDetails, linear_algebra::{add_vector, matrix_cross_vector}};
mod neural_network_layer;
mod activation_functions;
mod layer_details;
mod cost_functions;
mod neural_network;
mod linear_algebra;
fn main() {
  let neural_network = NeuralNetwork::new(&[
    LayerDetails::new(2,activation_functions::Linear),
    LayerDetails::new(10,activation_functions::Linear),
    LayerDetails::new(1,activation_functions::Linear)
  ], cost_functions::MSE);
  let predicted = neural_network.forward_propagate(vec![1.0,2.0]);
  // println!("{:#?}", neural_network);
  println!("{:?}",predicted);

  // let matrix = vec![
  //   vec![1,2,3,4],
  //   vec![5,6,7,8],
  // ];
  // let vec = vec![99,8,7,2];
  // println!("{:?}",matrix_cross_vector(&matrix, &vec));
}
