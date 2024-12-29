use crate::{
  layer_details::LayerDetails,
  // linear_algebra::{add_vector, matrix_dot_vector},
};
use activation_functions::{ActivationFunction, Linear};
use cost_functions::{CostFunction, MSE};
use neural_network::NeuralNetwork;
mod activation_functions;
mod cost_functions;
mod layer_details;
mod linear_algebra;
mod neural_network;
mod neural_network_layer;
mod stat;
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
  let mut input = vec![vec![],vec![]];
  let mut output = vec![];
  for i in 1..100 {
    for j in 1..100 {
      let i = i as f64;
      let j = j as f64;
      input[0].push(i);
      input[1].push(j);
      output.push(i + j);
    }
  }
  let output = vec![output];

  // let output_mean: f64 = output.iter().sum::<f64>() / output.len() as f64;
  // let output_sd = (output.iter().map(|data| data.powi(2)).sum::<f64>() / output.len() as f64
  //   - output_mean.powi(2))
  // .sqrt();

  // output = output
  //   .iter()
  //   .map(|data| (*data - output_mean) / output_sd)
  //   .collect::<Vec<f64>>();

  // for (x, y) in input.iter().zip(output) {
  //   let predicted = neural_network.forward_propagate(x);
  //   if predicted[0].is_nan() {
  //     break;
  //   }
  //   println!("x: {:?}, y: {:?}, predicted: {:?}", x, y, predicted);
  //   neural_network.back_propagate(&vec![y], learning_rate);
  // }

  // println!("{:#?}", neural_network);

  // let test = vec![
  //   (8.0 - input_mean) / input_sd,
  //   (14.0 - input_mean) / input_sd,
  // ];
  // let predicted = neural_network.forward_propagate(&test);
  // println!("result: {:?}", predicted[0] * output_sd + output_mean);

  neural_network.train(input, output, learning_rate);

  println!("{:?}",neural_network.predict(vec![100.5,15.0]));
}
