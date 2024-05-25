use std::fmt::Debug;
pub trait ActivationFunction: Debug + Clone{
  /// apply g function to z (g(z))
  fn activate(value: &Vec<f64>) -> Vec<f64>;

  /// apply g' function to z relativee to z (g'(z) or dg(z)/dz)
  fn derivative(value: &Vec<f64>) -> Vec<f64>;
}

#[derive(Debug,Clone)]
pub struct Linear;

impl ActivationFunction for Linear{
  fn activate(value: &Vec<f64>) -> Vec<f64>{
    value.iter().cloned().collect()
  }
  fn derivative(value: &Vec<f64>) -> Vec<f64>{
    vec![1.0;value.len()]
  }
}