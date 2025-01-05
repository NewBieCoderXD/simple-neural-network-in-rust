use crate::linear_algebra::{
  add_matrices, add_vector, get_row, matrix_dot_vector, minus_vector,
  scalar_multiply_matrix, scalar_multiply_vector, transpose_matrix, vector_dot_transposed_vector,
};
use crate::stat::mean_and_standard_deviation;
use crate::{
  cost_functions::CostFunction, hadamard_product, layer_details::LayerDetails,
  neural_network_layer::NeuralNetworkLayer,
};
use rand::Rng;
use std::fmt::Debug;

#[derive(Debug)]
pub struct NeuralNetwork {
  pub layers: Vec<NeuralNetworkLayer>,
  cost_function: CostFunction,
  x_mean: Option<Vec<f64>>,
  x_sd: Option<Vec<f64>>,
  y_mean: Option<Vec<f64>>,
  y_sd: Option<Vec<f64>>,
}

impl NeuralNetwork {
  // Xavier Initialization
  fn random_xavier(previous_size: usize, current_size: usize) -> f64 {
    let x = f64::sqrt(6.0 / (current_size + previous_size) as f64);
    return rand::thread_rng().gen_range(-x..x);
  }
  pub fn new(layers_details: &[LayerDetails], cost_function: CostFunction) -> NeuralNetwork {
    assert!(
      layers_details.len() >= 2,
      "There are not enough layers to function({} layers), please make at least 2",
      layers_details.len()
    );
    let mut layers: Vec<NeuralNetworkLayer> = vec![];
    let layers_count = layers_details.len();
    layers.reserve(layers_count);

    let mut network = NeuralNetwork {
      layers: vec![],
      cost_function: cost_function,
      x_mean: None,
      x_sd: None,
      y_mean: None,
      y_sd: None,
    };

    for (index, layer_details) in layers_details.iter().enumerate() {
      let weights;
      let biases;
      if index != 0 {
        let curr_layer_size = layer_details.layer_size;
        let prev_layer_size = layers[index - 1].size;
        weights = (0..curr_layer_size)
          .map(|_| {
            (0..prev_layer_size)
              .map(|_| Self::random_xavier(prev_layer_size, curr_layer_size))
              .collect()
          })
          .collect();
        biases = vec![0.0; layer_details.layer_size];
      } else {
        weights = vec![];
        biases = vec![];
      }
      let current_layer: NeuralNetworkLayer = NeuralNetworkLayer {
        biases,
        weights,
        size: layer_details.layer_size,
        activation_function: layer_details.activation_function,
        activations: None,
        z_values: None,
      };

      layers.push(current_layer);
    }

    network.layers = layers;

    return network;
  }

  pub fn forward_propagate(&mut self, input: &Vec<f64>) -> Vec<f64> {
    assert_eq!(input.len(), self.layers[0].size);
    self.layers[0].activations = Some(input.clone());
    for index in 1..self.layers.len() {
      let (left_splitted, right_splitted) = self.layers.split_at_mut(index);
      let curr_layer = &mut right_splitted[0];
      let prev_layer = &left_splitted[index - 1];
      let activations = prev_layer.activations.as_ref().unwrap();
      let z_values = add_vector(
        &matrix_dot_vector(&curr_layer.weights, &activations),
        &curr_layer.biases,
      );
      curr_layer.activations = Some(curr_layer.activation_function.activate(&z_values));
      curr_layer.z_values = Some(z_values.clone());
    }
    return self.layers.last().unwrap().activations.clone().unwrap();
  }

  pub fn back_propagate(&mut self, actual: &Vec<f64>, learning_rate: f64) {
    let last_layer = self.layers.last().unwrap();
    assert_eq!(actual.len(), last_layer.size);

    let z_values = last_layer.z_values.as_ref().unwrap();
    let activations = last_layer.activations.as_ref().unwrap();
    let cost_derivative = self.cost_function.derivative(&activations, actual);
    let activation_derivative = last_layer.activation_function.derivative(&z_values);
    let mut error = hadamard_product!(cost_derivative, activation_derivative);

    for i in (1..self.layers.len()).rev() {
      let (left_splitted, right_splitted) = self.layers.split_at_mut(i);
      let curr_layer = &mut right_splitted[0];
      let left_layer = &left_splitted[i - 1];

      Self::back_propagate_weights(&error, curr_layer, left_layer, learning_rate);
      Self::back_propagate_biases(&error, curr_layer, learning_rate);

      if i > 1 {
        error = hadamard_product!(
          left_layer
            .activation_function
            .derivative(left_layer.z_values.as_ref().unwrap()),
          matrix_dot_vector(&transpose_matrix(&curr_layer.weights), &error)
        );
      }
    }
  }

  pub fn train(&mut self, x: Vec<Vec<f64>>, y: Vec<Vec<f64>>, learning_rate: f64) {
    let (x_mean, x_sd): (Vec<f64>, Vec<f64>) = x
      .iter()
      .map(|series| mean_and_standard_deviation(series))
      .unzip();
    let (y_mean, y_sd): (Vec<f64>, Vec<f64>) = y
      .iter()
      .map(|series| mean_and_standard_deviation(series))
      .unzip();
    self.x_mean = Some(x_mean);
    self.x_sd = Some(x_sd);
    self.y_mean = Some(y_mean);
    self.y_sd = Some(y_sd);

    // println!("x_mean: {:?}, x_sd: {:?}, y_mean: {:?}, y_sd: {:?}",self.x_mean,self.x_sd,self.y_mean,self.y_sd);

    for row in 0..x[0].len() {
      let x_of_row = x.iter().map(|series| series[row]).collect::<Vec<f64>>();
      let y_of_row = y.iter().map(|series| series[row]).collect::<Vec<f64>>();
      let predicted = self.forward_propagate(&Self::vec_standard_scale(
        &x_of_row,
        &self.x_mean.as_ref().unwrap(),
        &self.x_sd.as_ref().unwrap(),
      ));

      let scaled_y = Self::vec_standard_scale(
        &y_of_row,
        self.y_mean.as_ref().unwrap(),
        self.y_sd.as_ref().unwrap(),
      );
      let unscaled_predicted = Self::vec_standard_unscale(
        &predicted,
        self.y_mean.as_ref().unwrap(),
        self.y_sd.as_ref().unwrap(),
      );
      if row % (x[0].len()/10) == 0 {
        let error = self.calculate_error(&y_of_row, &unscaled_predicted);
        println!(
          "x: {:?}, y: {:?}, predicted: {:?}, error: {}, scaled_error: {}",
          x_of_row, y_of_row, unscaled_predicted, error, self.calculate_error(&scaled_y, &predicted)
        );
      }
      self.back_propagate(&scaled_y, learning_rate);
    }
  }

  pub fn predict(&mut self, x: &Vec<f64>) -> Vec<f64> {
    let predicted = self.forward_propagate(&Self::vec_standard_scale(
      &x,
      &self.x_mean.as_ref().unwrap(),
      &self.x_sd.as_ref().unwrap(),
    ));
    return Self::vec_standard_unscale(
      &predicted,
      self.y_mean.as_ref().unwrap(),
      self.y_sd.as_ref().unwrap(),
    );
  }

  fn standard_scale(value: f64, mean: f64, sd: f64) -> f64 {
    return (value - mean) / sd;
  }

  fn standard_unscale(value: f64, mean: f64, sd: f64) -> f64 {
    return value * sd + mean;
  }

  fn vec_standard_scale(x: &Vec<f64>, mean: &Vec<f64>, sd: &Vec<f64>) -> Vec<f64> {
    return x
      .iter()
      .zip(mean)
      .zip(sd)
      .map(|((&value, &mean), &sd)| Self::standard_scale(value, mean, sd))
      .collect::<Vec<f64>>();
  }

  fn vec_standard_unscale(x: &Vec<f64>, mean: &Vec<f64>, sd: &Vec<f64>) -> Vec<f64> {
    return x
      .iter()
      .zip(mean)
      .zip(sd)
      .map(|((&value, &mean), &sd)| Self::standard_unscale(value, mean, sd))
      .collect::<Vec<f64>>();
  }

  fn back_propagate_weights(
    error: &Vec<f64>,
    curr_layer: &mut NeuralNetworkLayer,
    prev_layer: &NeuralNetworkLayer,
    learning_rate: f64,
  ) {
    let weight_derivative =
      vector_dot_transposed_vector(&error, &prev_layer.activations.as_ref().unwrap());

    curr_layer.weights = add_matrices(
      &curr_layer.weights,
      &scalar_multiply_matrix(-learning_rate, &weight_derivative),
    );
  }

  fn back_propagate_biases(
    error: &Vec<f64>,
    curr_layer: &mut NeuralNetworkLayer,
    learning_rate: f64,
  ) {
    curr_layer.biases = add_vector(
      &curr_layer.biases,
      &scalar_multiply_vector(-learning_rate, &error),
    );
  }

  pub fn calculate_error(&self, actual: &Vec<f64>, predicted: &Vec<f64>) -> f64 {
    return self.cost_function.cost(actual, predicted);
  }

  pub fn test(&mut self, input: &Vec<Vec<f64>>, output: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut mae = vec![0.0; output.len()];
    for row_index in 0..input[0].len() {
      let test_sample = get_row(input, row_index);
      let predicted = self.predict(&test_sample);

      let error = add_vector(&predicted, &minus_vector(&get_row(output, row_index))).iter().map(|x|x.abs()).collect();
      mae = add_vector(
        &mae,
        &error
      );

        println!(
          "input: {input:?}, predicted: {predicted:?}, actual: {actual:?}, error: {error:?}",
          input = get_row(input, row_index),
          actual = get_row(output, row_index)
        );
    }
    return scalar_multiply_vector(1.0 / output[0].len() as f64, &mae);
    // let mut rss = vec![0.0; output.len()];
    // for row_index in 0..input[0].len() {
    //   let test_sample = get_row(input, row_index);
    //   let predicted = self.predict(&test_sample);

    //   let curr_rss = powi_vector(
    //     &add_vector(&get_row(output, row_index), &minus_vector(&predicted)),
    //     2,
    //   );
    //   rss = add_vector(&rss, &curr_rss);

    //   println!(
    //     "input: {input:?}, predicted: {predicted:?}, actual: {actual:?}, RSS: {curr_rss:?}",
    //     input = get_row(input, row_index),
    //     actual = get_row(output, row_index)
    //   );
    // }

    // let tss = output
    //   .iter()
    //   .map(|column| {
    //     let sd = mean_and_standard_deviation(column).1;
    //     let tss = sd.powi(2) * (column.len() - 1) as f64;
    //     return tss;
    //   })
    //   .collect::<Vec<f64>>();
    // println!("tss: {:?}",tss);
    // return add_vector(
    //   &vec![1.0; output.len()],
    //   &minus_vector(&hadamard_product!(&rss, powi_vector(&tss, -1))),
    // );
  }
}
