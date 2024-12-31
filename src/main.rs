use crate::layer_details::LayerDetails;
use neural_network::NeuralNetwork;
use rand::Rng;
mod activation_functions;
mod cost_functions;
mod layer_details;
mod linear_algebra;
mod neural_network;
mod neural_network_layer;
mod stat;

fn main() {
  let learning_rate = 0.005;
  let scale: f64 = 50.0;
  let mut neural_network = NeuralNetwork::new(
    &[
      LayerDetails::new(2, activation_functions::Linear),
      // LayerDetails::new(10, activation_functions::Linear),
      LayerDetails::new(1, activation_functions::Linear),
    ],
    cost_functions::MSE,
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

  for _ in 0..5 {
    let i = rng.gen::<f64>() * 100.0 * scale * if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
    let j = rng.gen::<f64>() * scale * if rng.gen_bool(0.5) { 1.0 } else { -1.0 };

    let predicted = neural_network.predict(&vec![i, j]);

    println!(
      "lhs: {i}, rhs: {j}, predicted: {predicted}, actual: {actual}, error: {error}",
      predicted = predicted[0],
      actual = i + j,
      error = neural_network.calculate_error(&vec![i + j], &predicted)
    );
  }
}
