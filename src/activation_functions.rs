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
    value.iter().map(|&ele| ele.max(0.0)).collect()
  }
  fn derivative(value: &Vec<f64>) -> Vec<f64>{
    value.iter().map(|&ele| {
      if ele<=0.0{
        return 0.0;
      }
      return 1.0;
    }).collect()
  }
}