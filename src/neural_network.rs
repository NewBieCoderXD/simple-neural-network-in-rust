use crate::linear_algebra::{
  add_matrices, add_vector, matrix_dot_vector, scalar_multiply_matrix, scalar_multiply_vector,
  transpose_matrix, vector_dot_transposed_vector,
};
use crate::stat::mean_and_standard_deviation;
use crate::{
  activation_functions::ActivationFunction, cost_functions::CostFunction, hadamard_product,
  layer_details::LayerDetails, neural_network_layer::NeuralNetworkLayer,
};
use rand::Rng;
use std::fmt::Debug;

#[derive(Debug)]
pub struct NeuralNetwork<T: ActivationFunction, C: CostFunction> {
  pub layers: Vec<NeuralNetworkLayer<T>>,
  cost_function: C,
  x_mean: Option<Vec<f64>>,
  x_sd: Option<Vec<f64>>,
  y_mean: Option<Vec<f64>>,
  y_sd: Option<Vec<f64>>,
}

impl<T: ActivationFunction, C: CostFunction> NeuralNetwork<T, C> {
  // Xavier Initialization
  fn random_vector(previous_size: usize, current_size: usize) -> Vec<f64> {
    let x = f64::sqrt(6.0 / (current_size + previous_size) as f64);
    (0..previous_size)
      .map(|_index| rand::thread_rng().gen_range(-x..x))
      .collect()
  }
  pub fn new(layers_details: &[LayerDetails<T>], cost_function: C) -> NeuralNetwork<T, C> {
    assert!(
      layers_details.len() >= 2,
      "There are not enough layers to function({} layers), please make at least 2",
      layers_details.len()
    );
    let mut layers: Vec<NeuralNetworkLayer<T>> = vec![];
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
        weights = (0..layer_details.layer_size)
          .map(|_| Self::random_vector(layers[index - 1].size, layer_details.layer_size))
          .collect();
        biases = vec![0.0; layer_details.layer_size];
      } else {
        weights = vec![];
        biases = vec![];
      }
      let current_layer: NeuralNetworkLayer<T> = NeuralNetworkLayer {
        biases,
        weights,
        size: layer_details.layer_size,
        // network: &network,
        // index: index,
        activation_function: layer_details.activation_function.clone(),
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
    // let mut activations = input;
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
      curr_layer.activations = Some(T::activate(&z_values));
      curr_layer.z_values = Some(z_values.clone());
    }
    return T::activate(&self.layers.last().unwrap().z_values.clone().unwrap());
  }

  pub fn back_propagate(&mut self, actual: &Vec<f64>, learning_rate: f64) {
    let last_layer = self.layers.last().unwrap();
    assert_eq!(actual.len(), last_layer.size);

    let z_values = last_layer.z_values.as_ref().unwrap();
    let activations = last_layer.activations.as_ref().unwrap();
    let cost_derivative = C::derivative(&activations, actual);
    let activation_derivative = T::derivative(&z_values);
    let mut error = hadamard_product!(cost_derivative, activation_derivative);

    let cost = C::cost(actual, activations.as_ref());
    println!("cost: {}", cost);
    println!("cost_derivative: {:?}", cost_derivative);

    for i in (1..self.layers.len()).rev() {
      let (left_splitted, right_splitted) = self.layers.split_at_mut(i);
      let curr_layer = &mut right_splitted[0];
      let left_layer = &left_splitted[i - 1];

      Self::back_propagate_weights(&error, curr_layer, left_layer, learning_rate);
      Self::back_propagate_biases(&error, curr_layer, learning_rate);

      if i > 1 {
        error = hadamard_product!(
          T::derivative(left_layer.z_values.as_ref().unwrap()),
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

    println!("{:?}", self.x_mean);

    for row in 0..x[0].len() {
      let x_of_row = x.iter().map(|series| series[row]).collect::<Vec<f64>>();
      let y_of_row = y.iter().map(|series| series[row]).collect::<Vec<f64>>();
      let predicted = self.forward_propagate(&Self::vec_standard_scale(
        &x_of_row,
        &self.x_mean.as_ref().unwrap(),
        &self.x_sd.as_ref().unwrap(),
      ));
      // if predicted[0].is_nan() {
      //   break;
      // }
      let unscaled_y = Self::vec_standard_scale(
        &y_of_row,
        self.y_mean.as_ref().unwrap(),
        self.y_sd.as_ref().unwrap(),
      );
      println!(
        "x: {:?}, y: {:?}, predicted: {:?}",
        x_of_row,
        y_of_row,
        Self::vec_standard_unscale(
          &predicted,
          self.y_mean.as_ref().unwrap(),
          self.y_sd.as_ref().unwrap()
        )
      );
      self.back_propagate(&unscaled_y, learning_rate);
    }
  }

  pub fn predict(&mut self, x: Vec<f64>) -> Vec<f64> {
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
    curr_layer: &mut NeuralNetworkLayer<T>,
    prev_layer: &NeuralNetworkLayer<T>,
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
    curr_layer: &mut NeuralNetworkLayer<T>,
    learning_rate: f64,
  ) {
    curr_layer.biases = add_vector(
      &curr_layer.biases,
      &scalar_multiply_vector(-learning_rate, &error),
    );
  }
}
