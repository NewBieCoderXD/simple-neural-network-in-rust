use crate::{
  layer_details::LayerDetails,
  // linear_algebra::{add_vector, matrix_dot_vector},
};
use neural_network::NeuralNetwork;
mod activation_functions;
mod cost_functions;
mod layer_details;
mod linear_algebra;
mod neural_network;
mod neural_network_layer;
fn main() {
  let learning_rate = 0.001;
  let mut neural_network = NeuralNetwork::new(
    &[
      LayerDetails::new(2, activation_functions::Linear),
      LayerDetails::new(10, activation_functions::Linear),
      LayerDetails::new(1, activation_functions::Linear),
    ],
    cost_functions::MSE,
  );

  println!("{:#?}", neural_network);

  // let datas = vec![
  //   (vec![1.0,2.0],vec![3.0])
  // ]
  let mut datas = vec![];
  for i in 1 .. 10{
    for j in 1 .. 10{
      let i = i as f64;
      let j = j as f64;
      datas.push((vec![i,j],vec![i+j]));
    }
  }

  for (x,y) in &datas{
    let predicted = neural_network.forward_propagate(x);
    if predicted[0].is_nan(){
      break;
    }
    println!("x: {:?}, y: {:?}, predicted: {:?}", x, y, predicted);
    // println!("{:?}", neural_network);
    neural_network.back_propagate(y, learning_rate);
  }
  
  println!("{:#?}", neural_network);

  let test = vec![8.0,14.0];
  let predicted = neural_network.forward_propagate(&test);
  println!("test: {:?}, result: {:?}", test, predicted);
}
