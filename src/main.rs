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
  let neural_network = NeuralNetwork::new(
    &[
      LayerDetails::new(2, activation_functions::Linear),
      LayerDetails::new(10, activation_functions::Linear),
      LayerDetails::new(1, activation_functions::Linear),
    ],
    cost_functions::MSE,
  );
  // let datas = vec![
  //   (vec![1.0,2.0],vec![3.0])
  // ]
  let mut datas = vec![];
  for i in 0 .. 10{
    for j in 0 .. 10{
      let i = i as f64;
      let j = j as f64;
      datas.push((vec![i,j],vec![i+j]));
    }
  }
  for (x,y) in &datas{
    let predicted = neural_network.forward_propagate(x);
    println!("x: {:?}, y: {:?}, predicted: {:?}", x, y, predicted);
    neural_network.back_propagate(y, learning_rate);
  }
  println!("{:#?}", neural_network);
  let predicted = neural_network.forward_propagate(&vec![8.0,14.0]);
  println!("{:?}", predicted);
  // let predicted2 = neural_network.forward_propagate(vec![1.0, 2.0]);
  // println!("{:?}", predicted2);

  // println!("{:#?}", neural_network);
  // let matrix = vec![
  //   vec![1,2,3,4],
  //   vec![5,6,7,8],
  // ];
  // let vec = vec![99,8,7,2];
  // println!("{:?}",matrix_dot_vector(&matrix, &vec));
}
