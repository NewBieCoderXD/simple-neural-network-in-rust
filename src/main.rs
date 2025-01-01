use crate::layer_details::LayerDetails;
use activation_functions::ActivationFunction;
use cost_functions::CostFunction;
use neural_network::NeuralNetwork;
use rand::Rng;
mod activation_functions;
mod cost_functions;
mod layer_details;
mod linear_algebra;
mod neural_network;
mod neural_network_layer;
mod stat;

fn adding() {
  let learning_rate = 0.005;
  let scale: f64 = 50.0;
  let mut neural_network = NeuralNetwork::new(
    &[
      LayerDetails::new(2, ActivationFunction::Linear),
      LayerDetails::new(1, ActivationFunction::Linear),
    ],
    CostFunction::MeanSquaredError,
  );

  println!("{:#?}", neural_network);

  let mut rng = rand::thread_rng();

  let mut input = vec![vec![], vec![]];
  let mut output = vec![];

  for _ in 1..100000 {
    let i = rng.gen::<f64>() * scale;
    let j = rng.gen::<f64>() * scale;
    input[0].push(i);
    input[1].push(j);
    output.push(i + j);
  }
  let output = vec![output];

  neural_network.train(input, output, learning_rate);

  let mut input = vec![vec![], vec![]];
  let mut output = vec![];
  for _ in 0..100 {
    let i = rng.gen::<f64>() * scale * 100.0;
    let j = rng.gen::<f64>() * scale * -1.0;
    input[0].push(i);
    input[1].push(j);
    output.push(i + j);
  }
  let output = vec![output];
  println!("R2: {:?}", neural_network.test(&input, &output));
}

fn multiplying() {
  let learning_rate = 0.01;
  let scale: f64 = 50.0;
  let mut neural_network = NeuralNetwork::new(
    &[
      LayerDetails::new(2, ActivationFunction::Linear),
      LayerDetails::new(10, ActivationFunction::ReLU),
      LayerDetails::new(10, ActivationFunction::Sigmoid),
      LayerDetails::new(10, ActivationFunction::Tanh),
      LayerDetails::new(1, ActivationFunction::Linear),
    ],
    CostFunction::MeanSquaredError,
  );

  println!("{:#?}", neural_network);

  let mut rng = rand::thread_rng();

  let mut input = vec![vec![], vec![]];
  let mut output = vec![];

  for _ in 1..100000 {
    let i = rng.gen::<f64>() * scale - scale / 2.0;
    let j = rng.gen::<f64>() * scale - scale / 2.0;
    input[0].push(i);
    input[1].push(j);
    output.push(i * j);
  }
  let output = vec![output];

  neural_network.train(input, output, learning_rate);

  let mut input = vec![vec![], vec![]];
  let mut output = vec![];
  for _ in 0..100 {
    let i = rng.gen::<f64>() * scale - scale / 2.0;
    let j = rng.gen::<f64>() * scale - scale / 2.0;
    input[0].push(i);
    input[1].push(j);
    output.push(i * j);
  }
  let output = vec![output];
  println!("R2: {:?}", neural_network.test(&input, &output));
}

fn main() {
  multiplying();
}
